"""Dual-head Stage 1 model.

The Stage 1 model adapts a causal language-model backbone with LoRA and
exposes two prediction heads from a shared representation:

* **Generation head**: the backbone's standard causal-LM head. It produces
  the documented KTAS adjudication sequence ``(a, m, s, d, l)`` (or just
  the final level ``l`` in the Triage-level ablation) token by token after
  the ``[KTAS sequence]`` delimiter.
* **Classification head**: a three-layer MLP (``hidden_size -> 512 -> 256
  -> 5``) attached to the hidden state at the delimiter position. It
  provides an independent ordinal prediction of the final triage level.

The same module covers all four Stage 1 ablations through the
``use_generation_head`` / ``use_classification_head`` flags.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class KTASUnifiedModel(nn.Module):
    """Dual-head model wrapping a causal-LM backbone.

    Parameters
    ----------
    base_model:
        A ``transformers`` causal LM (potentially LoRA-adapted) that returns
        hidden states when ``output_hidden_states=True``.
    hidden_size:
        Backbone hidden size used by the classification MLP.
    num_classes:
        Number of ordinal classes (5 for KTAS levels).
    cls_hidden_dim:
        Width of the first MLP hidden layer (the second layer is
        ``cls_hidden_dim // 2``).
    cls_dropout:
        Dropout rate inside the classification MLP.
    cls_dtype:
        Dtype of the MLP parameters; defaults to ``bfloat16`` to match the
        backbone in the paper configuration.
    use_generation_head:
        When ``False``, the model skips the generation loss entirely
        (Classification-only ablation).
    use_classification_head:
        When ``False``, the model only computes the generation loss
        (Triage-level and Triage-full context ablations).
    """

    def __init__(
        self,
        base_model: nn.Module,
        hidden_size: int,
        num_classes: int = 5,
        cls_hidden_dim: int = 512,
        cls_dropout: float = 0.1,
        cls_dtype: torch.dtype = torch.bfloat16,
        use_generation_head: bool = True,
        use_classification_head: bool = True,
    ) -> None:
        super().__init__()
        if not (use_generation_head or use_classification_head):
            raise ValueError(
                "At least one of use_generation_head or use_classification_head "
                "must be True."
            )

        self.base_model = base_model
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.use_generation_head = use_generation_head
        self.use_classification_head = use_classification_head

        if use_classification_head:
            self.severity_classifier = nn.Sequential(
                nn.Linear(hidden_size, cls_hidden_dim),
                nn.LayerNorm(cls_hidden_dim),
                nn.GELU(),
                nn.Dropout(cls_dropout),
                nn.Linear(cls_hidden_dim, cls_hidden_dim // 2),
                nn.LayerNorm(cls_hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(cls_dropout),
                nn.Linear(cls_hidden_dim // 2, num_classes),
            )
            self._init_classifier_weights()
            self.severity_classifier = self.severity_classifier.to(dtype=cls_dtype)
        else:
            self.severity_classifier = None

    # ----------------------------------------------------------- Initialization

    def _init_classifier_weights(self) -> None:
        for module in self.severity_classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ----------------------------------------------------------- Forward

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        cls_positions: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run the backbone and any active heads.

        Parameters
        ----------
        input_ids, attention_mask:
            Standard tokenized batch.
        labels:
            Causal-LM labels with prompt tokens masked to ``-100``. Required
            only when ``use_generation_head`` is True.
        cls_positions:
            Long tensor of shape ``(batch,)`` giving the per-example
            position of the last prompt token (the final token of the
            ``[KTAS sequence]`` delimiter). Required when
            ``use_classification_head`` is True.
        """
        need_hidden_states = self.use_classification_head

        if not self.use_generation_head:
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=None,
                output_hidden_states=True,
            )
            gen_loss = None
        else:
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=need_hidden_states,
            )
            gen_loss = outputs.loss

        result: Dict[str, torch.Tensor] = {
            "gen_loss": gen_loss,
            "logits": outputs.logits,
        }

        if self.use_classification_head:
            if cls_positions is None:
                raise ValueError(
                    "cls_positions is required when use_classification_head is True."
                )

            hidden_states = outputs.hidden_states[-1]
            batch_size, seq_len, _ = hidden_states.shape
            cls_positions = cls_positions.clamp(0, seq_len - 1)

            row_index = torch.arange(batch_size, device=hidden_states.device)
            cls_hidden = hidden_states[row_index, cls_positions]

            mlp_dtype = next(self.severity_classifier.parameters()).dtype
            if cls_hidden.dtype != mlp_dtype:
                cls_hidden = cls_hidden.to(dtype=mlp_dtype)

            result["severity_logits"] = self.severity_classifier(cls_hidden)
            result["cls_hidden"] = cls_hidden

        return result
