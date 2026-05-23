#!/usr/bin/env python3
"""Stage 2 trainer for Dr.KTAS — Guideline-Informed Modifier Re-adjudication.

Fine-tunes a Ministral-8B LoRA adapter that, given the Stage 1 head-
disagreement context and a list of guideline-aligned modifier candidates,
outputs a JSON object whose ``"선택"`` field identifies the candidate. The
training distribution is produced by ``scripts/stage2_prepare_data.py`` and
saved as JSONL chat records of the form::

    {"messages": [
        {"role": "user",      "content": "<prompt with candidates>"},
        {"role": "assistant", "content": "{...,\"선택\": \"N\",\"이유\": \"...\"}"}
    ]}

Two loss-masking strategies are supported, matching the calibration ablation
discussed in Section III-E of the paper:

* ``selection_only``  — Only the selected candidate-number tokens contribute
  to the loss. Other JSON tokens are masked with ``-100``.
* ``json_structure``  — The JSON header up to the ``"이유"`` value is
  supervised; only the free-text reason is masked. This is the default,
  matching ``loss_mode='json_structure'`` in the released configuration.

Defaults are aligned with paper Table III (1 epoch, lr 1.5e-5, effective
batch 16, linear schedule with warmup 0.1, weight decay 0.02, max gradient
norm 0.5, max sequence length 4,096, greedy decoding at inference).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import Dataset
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)
import yaml


# Allow running directly from the repository root without installation.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("drktas.stage2_train")


# =============================================================================
# Selection / reason marker location in the JSON response
# =============================================================================

SELECTION_START_MARKER = '"선택": "'
SELECTION_END_MARKER = '", "이유"'
REASON_VALUE_MARKER = '"이유": "'


def _char_to_token_index(
    token_ids: List[int],
    tokenizer,
    char_position: int,
) -> int:
    """Smallest token index whose decoded prefix reaches ``char_position``.

    Implemented via binary search; the response is short (a few dozen
    tokens) so the log-factor cost is negligible.
    """
    lo, hi = 1, len(token_ids)
    while lo < hi:
        mid = (lo + hi) // 2
        decoded = tokenizer.decode(token_ids[:mid], skip_special_tokens=True)
        if len(decoded) >= char_position:
            hi = mid
        else:
            lo = mid + 1
    return lo


def find_selection_token_range(
    assistant_token_ids: List[int],
    tokenizer,
    response_text: str,
) -> Tuple[int, int]:
    """Return ``(start, end)`` token indices for the candidate-number value.

    The range is given relative to ``assistant_token_ids``. When either of
    the surrounding markers cannot be located the function falls back to
    ``(0, len(assistant_token_ids))``, which leaves the entire assistant
    span supervised.
    """
    marker_pos = response_text.find(SELECTION_START_MARKER)
    if marker_pos == -1:
        return 0, len(assistant_token_ids)
    selection_char_start = marker_pos + len(SELECTION_START_MARKER)

    end_pos = response_text.find(SELECTION_END_MARKER, selection_char_start)
    if end_pos == -1:
        return 0, len(assistant_token_ids)
    selection_char_end = end_pos

    start_tok = _char_to_token_index(assistant_token_ids, tokenizer, selection_char_start)
    end_tok = _char_to_token_index(assistant_token_ids, tokenizer, selection_char_end)
    if end_tok <= start_tok:
        end_tok = start_tok + 1
    return start_tok - 1, end_tok


def find_reason_start_token_index(
    assistant_token_ids: List[int],
    tokenizer,
    response_text: str,
) -> int:
    """Token index (assistant-relative) at which the reason value starts.

    Returns ``len(assistant_token_ids)`` if the reason marker cannot be
    found, which disables reason-masking for that example.
    """
    marker_pos = response_text.find(REASON_VALUE_MARKER)
    if marker_pos == -1:
        return len(assistant_token_ids)
    reason_char_start = marker_pos + len(REASON_VALUE_MARKER)
    tok = _char_to_token_index(assistant_token_ids, tokenizer, reason_char_start)
    return tok - 1


# =============================================================================
# Data collator
# =============================================================================

class ChatDataCollatorWithLossMasking:
    """Tokenize chat-format records and produce label masks.

    The collator delegates tokenization to the model's chat template so the
    user/assistant turn boundaries match what the LoRA-adapted backbone
    expects at inference time. The ``loss_mode`` argument controls which
    spans of the assistant response contribute to the cross-entropy loss.
    """

    def __init__(
        self,
        tokenizer,
        max_length: int = 4096,
        loss_mode: str = "json_structure",
    ) -> None:
        if loss_mode not in {"selection_only", "json_structure"}:
            raise ValueError(
                f"loss_mode must be 'selection_only' or 'json_structure', got {loss_mode!r}."
            )
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.loss_mode = loss_mode

    # ---------------------------------------------------------------- Helpers

    def _user_only_length(self, messages: List[Dict[str, str]]) -> int:
        user_token_ids = self.tokenizer.apply_chat_template(
            [messages[0]],
            tokenize=True,
            add_generation_prompt=True,
        )
        return len(user_token_ids)

    def _full_token_ids(self, messages: List[Dict[str, str]]) -> List[int]:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )

    # ---------------------------------------------------------------- Callable

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_input_ids: List[List[int]] = []
        batch_attention_mask: List[List[int]] = []
        batch_labels: List[List[int]] = []

        for example in examples:
            messages = example["messages"]
            assistant_content = messages[1]["content"]

            full_token_ids = self._full_token_ids(messages)
            user_len = self._user_only_length(messages)

            if len(full_token_ids) > self.max_length:
                full_token_ids = full_token_ids[: self.max_length]
            input_ids = full_token_ids
            attention_mask = [1] * len(input_ids)
            labels = [-100] * len(input_ids)

            assistant_token_ids = input_ids[user_len:]
            if assistant_token_ids:
                if self.loss_mode == "selection_only":
                    sel_start, sel_end = find_selection_token_range(
                        assistant_token_ids, self.tokenizer, assistant_content
                    )
                    abs_start = user_len + sel_start
                    abs_end = user_len + sel_end
                    for i in range(abs_start, min(abs_end, len(labels))):
                        labels[i] = input_ids[i]
                else:  # 'json_structure'
                    # Supervise the entire assistant span...
                    for i in range(user_len, len(labels)):
                        labels[i] = input_ids[i]
                    # ...then mask the free-text reason value.
                    reason_start = find_reason_start_token_index(
                        assistant_token_ids, self.tokenizer, assistant_content
                    )
                    abs_reason_start = user_len + reason_start
                    for i in range(abs_reason_start, len(labels)):
                        labels[i] = -100

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)

        max_len = min(max(len(ids) for ids in batch_input_ids), self.max_length)
        pad_id = self.tokenizer.pad_token_id

        padded_input_ids: List[List[int]] = []
        padded_attention_mask: List[List[int]] = []
        padded_labels: List[List[int]] = []
        for ids, mask, labels in zip(batch_input_ids, batch_attention_mask, batch_labels):
            pad = max_len - len(ids)
            padded_input_ids.append(ids + [pad_id] * pad)
            padded_attention_mask.append(mask + [0] * pad)
            padded_labels.append(labels + [-100] * pad)

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }


def report_loss_masking(
    collator: ChatDataCollatorWithLossMasking,
    dataset: Dataset,
    tokenizer,
    num_samples: int = 3,
) -> None:
    """Sanity-check the collator on the first few samples and log a summary."""
    if len(dataset) == 0:
        return
    logger.info("Loss-masking check (first %d samples):", num_samples)
    for i in range(min(num_samples, len(dataset))):
        example = dataset[i]
        batch = collator([example])
        input_ids = batch["input_ids"][0]
        labels = batch["labels"][0]
        kept = (labels != -100).sum().item()
        masked = (labels == -100).sum().item()
        kept_text = tokenizer.decode(
            [tid.item() for tid, lid in zip(input_ids, labels) if lid != -100],
            skip_special_tokens=True,
        )
        logger.info(
            "  sample %d: kept=%d masked=%d total=%d kept_text=%r",
            i,
            kept,
            masked,
            len(input_ids),
            kept_text[:120] + ("..." if len(kept_text) > 120 else ""),
        )


# =============================================================================
# Data / model loading
# =============================================================================

def load_chat_dataset(path: str | Path) -> Dataset:
    """Load a JSONL file produced by ``stage2_prepare_data.py``."""
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    dataset = Dataset.from_list(rows)
    logger.info("Loaded %d samples from %s", len(dataset), path)
    return dataset


def load_model_and_tokenizer(cfg: "Stage2Config"):
    logger.info("Loading backbone: %s", cfg.backbone)

    tokenizer = AutoTokenizer.from_pretrained(cfg.backbone, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if cfg.bf16 else torch.float16,
    }

    if cfg.quantization == "4bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
        )
    else:
        model_kwargs["device_map"] = "auto"

    if cfg.use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(cfg.backbone, **model_kwargs)

    if cfg.quantization == "4bit":
        model = prepare_model_for_kbit_training(model)

    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    logger.info("Backbone parameters: %s", f"{model.num_parameters():,}")
    return model, tokenizer


def apply_lora(model, cfg: "Stage2Config"):
    """Apply LoRA to all attention and FFN projection layers (Mistral)."""
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=target_modules,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "LoRA: r=%d alpha=%d dropout=%.2f trainable=%s / %s (%.2f%%)",
        cfg.lora_r,
        cfg.lora_alpha,
        cfg.lora_dropout,
        f"{trainable:,}",
        f"{total:,}",
        100 * trainable / max(total, 1),
    )
    return model


# =============================================================================
# Config
# =============================================================================

@dataclass
class Stage2Config:
    # Model
    backbone: str = "mistralai/Ministral-8B-Instruct-2410"
    quantization: str = "4bit"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    # LoRA
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05

    # Data
    train_data: str = ""
    val_data: str = ""
    max_seq_length: int = 4096

    # Loss
    loss_mode: str = "json_structure"

    # Training (paper Table III)
    output_dir: str = "runs/stage2"
    num_epochs: int = 1
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4  # effective batch = 4 * 4 = 16
    learning_rate: float = 1.5e-5
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "linear"
    weight_decay: float = 0.02
    max_grad_norm: float = 0.5

    # Saving / evaluation
    eval_strategy: str = "steps"
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 3
    early_stopping_patience: int = 5
    logging_steps: int = 10

    # Misc
    bf16: bool = True
    gradient_checkpointing: bool = True
    use_flash_attention: bool = True
    seed: int = 42
    dataloader_num_workers: int = 4
    resume_from_checkpoint: Optional[str] = None

    # ------------------------------------------------------------------ Loaders

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Stage2Config":
        cfg = cls()
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        def set_block(block: Dict[str, Any], remap: Dict[str, str]) -> None:
            for src, dst in remap.items():
                if src in block and block[src] is not None:
                    setattr(cfg, dst, block[src])

        set_block(
            data.get("model", {}),
            {
                "backbone": "backbone",
                "quantization": "quantization",
                "bnb_4bit_quant_type": "bnb_4bit_quant_type",
                "bnb_4bit_use_double_quant": "bnb_4bit_use_double_quant",
            },
        )
        set_block(
            data.get("lora", {}),
            {"r": "lora_r", "alpha": "lora_alpha", "dropout": "lora_dropout"},
        )
        set_block(
            data.get("data", {}),
            {"max_seq_length": "max_seq_length"},
        )
        set_block(
            data.get("loss", {}),
            {"loss_mode": "loss_mode"},
        )
        set_block(
            data.get("training", {}),
            {
                "num_epochs": "num_epochs",
                "per_device_train_batch_size": "per_device_train_batch_size",
                "per_device_eval_batch_size": "per_device_eval_batch_size",
                "gradient_accumulation_steps": "gradient_accumulation_steps",
                "learning_rate": "learning_rate",
                "warmup_ratio": "warmup_ratio",
                "lr_scheduler_type": "lr_scheduler_type",
                "weight_decay": "weight_decay",
                "max_grad_norm": "max_grad_norm",
                "seed": "seed",
            },
        )
        return cfg

    def apply_cli_overrides(self, args: argparse.Namespace) -> None:
        for key, value in vars(args).items():
            if value is None or key == "config":
                continue
            if hasattr(self, key):
                setattr(self, key, value)


# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2 trainer for Dr.KTAS")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--train_data", type=str, default=None)
    parser.add_argument("--val_data", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--loss_mode", type=str, default=None, choices=["selection_only", "json_structure"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Stage2Config.from_yaml(args.config)
    cfg.apply_cli_overrides(args)

    if not cfg.train_data or not cfg.val_data:
        raise SystemExit(
            "train_data and val_data are required. Pass --train_data / --val_data."
        )

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)

    logger.info("Stage 2 — Guideline-Informed Modifier Re-adjudication trainer")
    logger.info("Backbone: %s", cfg.backbone)
    logger.info("Loss mode: %s", cfg.loss_mode)
    logger.info(
        "LR=%s scheduler=%s warmup=%.2f wd=%.3f max_grad_norm=%.2f epochs=%d eff_batch=%d",
        cfg.learning_rate,
        cfg.lr_scheduler_type,
        cfg.warmup_ratio,
        cfg.weight_decay,
        cfg.max_grad_norm,
        cfg.num_epochs,
        cfg.per_device_train_batch_size
        * cfg.gradient_accumulation_steps
        * max(1, torch.cuda.device_count()),
    )

    # ---------------------------------------------------------- Data + model

    train_dataset = load_chat_dataset(cfg.train_data)
    val_dataset = load_chat_dataset(cfg.val_data)

    model, tokenizer = load_model_and_tokenizer(cfg)
    model = apply_lora(model, cfg)

    collator = ChatDataCollatorWithLossMasking(
        tokenizer=tokenizer,
        max_length=cfg.max_seq_length,
        loss_mode=cfg.loss_mode,
    )
    report_loss_masking(collator, train_dataset, tokenizer, num_samples=3)

    # ---------------------------------------------------------- Trainer

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        bf16=cfg.bf16,
        logging_steps=cfg.logging_steps,
        eval_strategy=cfg.eval_strategy,
        eval_steps=cfg.eval_steps,
        save_strategy=cfg.eval_strategy,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=cfg.seed,
        dataloader_num_workers=cfg.dataloader_num_workers,
        remove_unused_columns=False,
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs=(
            {"use_reentrant": False} if cfg.gradient_checkpointing else None
        ),
        ddp_find_unused_parameters=False,
        group_by_length=False,
    )

    callbacks = []
    if cfg.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        callbacks=callbacks,
    )

    train_result = trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)

    # ---------------------------------------------------------- Save

    final_adapter = output_dir / "final_adapter"
    model.save_pretrained(final_adapter)
    tokenizer.save_pretrained(final_adapter)
    logger.info("Saved final adapter to %s", final_adapter)

    metrics = dict(train_result.metrics)
    metrics["train_samples"] = len(train_dataset)
    metrics["val_samples"] = len(val_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    summary = {
        "backbone": cfg.backbone,
        "lora_r": cfg.lora_r,
        "lora_alpha": cfg.lora_alpha,
        "lora_dropout": cfg.lora_dropout,
        "loss_mode": cfg.loss_mode,
        "num_epochs": cfg.num_epochs,
        "learning_rate": cfg.learning_rate,
        "lr_scheduler_type": cfg.lr_scheduler_type,
        "warmup_ratio": cfg.warmup_ratio,
        "weight_decay": cfg.weight_decay,
        "max_grad_norm": cfg.max_grad_norm,
        "effective_batch_size": (
            cfg.per_device_train_batch_size
            * cfg.gradient_accumulation_steps
            * max(1, torch.cuda.device_count())
        ),
        "max_seq_length": cfg.max_seq_length,
        "bf16": cfg.bf16,
        "quantization": cfg.quantization,
        "use_flash_attention": cfg.use_flash_attention,
        "gradient_checkpointing": cfg.gradient_checkpointing,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "train_loss": metrics.get("train_loss"),
        "eval_loss": eval_metrics.get("eval_loss"),
        "seed": cfg.seed,
    }
    (output_dir / "training_config.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

    logger.info(
        "Stage 2 training complete. train_loss=%.4f eval_loss=%.4f",
        metrics.get("train_loss", float("nan")),
        eval_metrics.get("eval_loss", float("nan")),
    )


if __name__ == "__main__":
    main()
