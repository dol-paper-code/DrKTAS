#!/usr/bin/env python3
"""Stage 1 trainer for Dr.KTAS.

A single entry point that covers all four ablations reported in the paper:

    * Triage-level         (target_format='level',         use_classification_head=False)
    * Triage-full context  (target_format='full_sequence', use_classification_head=False)
    * Classification-only  (target_format='full_sequence', use_classification_head=True,
                            use_generation_head=False)
    * Dual-Head            (target_format='full_sequence', use_classification_head=True)

Behavior is controlled by a YAML config (`configs/stage1_*.yaml`). Any
config field can be overridden from the command line.

Launch examples
---------------

    # 4 GPUs, Dual-Head (paper's primary Stage 1 configuration)
    torchrun --nproc_per_node=4 scripts/stage1_train.py \
        --config configs/stage1_dual_head.yaml \
        --train_data path/to/your_train.csv \
        --output_dir runs/stage1_dual_head

    # Resume from the latest checkpoint
    torchrun --nproc_per_node=4 scripts/stage1_train.py \
        --config configs/stage1_dual_head.yaml \
        --train_data path/to/your_train.csv \
        --output_dir runs/stage1_dual_head \
        --resume_from_checkpoint runs/stage1_dual_head/latest_model
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import yaml
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

# Allow running directly from the repository root without installation.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from drktas.data_io import (  # noqa: E402
    KTAS_SEQUENCE_DELIMITER,
    build_response_text,
    compute_inverse_frequency_weights,
    extract_level_from_fullseverity,
)
from drktas.losses import (  # noqa: E402
    LossCombiner,
    UncertaintyWeighting,
    WeightedOrdinalCrossEntropy,
)
from drktas.models import KTASUnifiedModel  # noqa: E402


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("drktas.stage1_train")


# =============================================================================
# Distributed helpers
# =============================================================================

def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def get_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def safe_barrier() -> None:
    if dist.is_initialized():
        dist.barrier(device_ids=[int(os.environ.get("LOCAL_RANK", 0))])


# =============================================================================
# Tokenizer setup
# =============================================================================

def setup_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        if (
            tokenizer.unk_token is not None
            and tokenizer.unk_token_id != tokenizer.eos_token_id
        ):
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.pad_token_id = tokenizer.unk_token_id
        else:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    return tokenizer


# =============================================================================
# Dataset
# =============================================================================

class KTASDataset(Dataset):
    """Stage 1 dataset for all four ablations.

    Each CSV row provides a ``text`` field (the structured prompt up to
    and including the ``[KTAS sequence]`` delimiter) and a ``fullseverity``
    field (the documented KTAS sequence with the trailing level token).
    The response supervised by the generation head is determined by
    ``target_format``:

    * ``'full_sequence'``: the entire ``fullseverity`` string.
    * ``'level'``: only the trailing KTAS level digit (1-5).
    """

    def __init__(
        self,
        data_path: str | Path,
        tokenizer,
        max_length: int,
        target_format: str,
        max_samples: Optional[int] = None,
    ) -> None:
        if target_format not in ("level", "full_sequence"):
            raise ValueError(
                f"target_format must be 'level' or 'full_sequence', got {target_format!r}"
            )
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.target_format = target_format
        self.samples: List[Dict[str, Any]] = []

        path = Path(data_path)
        with path.open("r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if max_samples is not None and i >= max_samples:
                    break
                text = row.get("text", "").lstrip().rstrip("\r")
                fullseverity = row.get("fullseverity", "").strip()
                try:
                    response = build_response_text(fullseverity, target_format)
                except ValueError:
                    continue
                level = extract_level_from_fullseverity(fullseverity)
                if level is None:
                    continue
                self.samples.append(
                    {
                        "text": text,
                        "response": response,
                        "level": level - 1,  # 0-indexed for classification
                    }
                )

        if is_main_process():
            counter = Counter(s["level"] + 1 for s in self.samples)
            logger.info(
                "Loaded %d samples from %s (target_format=%s; level histogram=%s)",
                len(self.samples),
                path,
                target_format,
                dict(sorted(counter.items())),
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        text = sample["text"]
        response = sample["response"]
        level = sample["level"]

        if not text.endswith("\n"):
            text = text + "\n"

        full_text = text + response + self.tokenizer.eos_token
        full_encoding = self.tokenizer(
            full_text, add_special_tokens=True, return_tensors="pt", truncation=False
        )
        full_input_ids = full_encoding["input_ids"].squeeze(0)

        # Identify the response start position by re-tokenizing the suffix
        # without added special tokens; this is more robust than splitting on
        # the delimiter because BPE-style tokenizers can fuse boundary tokens.
        suffix_text = response + self.tokenizer.eos_token
        suffix_ids = self.tokenizer(
            suffix_text, add_special_tokens=False, return_tensors="pt", truncation=False
        )["input_ids"].squeeze(0)
        suffix_length = len(suffix_ids)
        full_length = len(full_input_ids)
        text_end_position = full_length - suffix_length

        if text_end_position > 0:
            actual_suffix = full_input_ids[text_end_position:]
            if not torch.equal(actual_suffix, suffix_ids):
                # Fallback: locate the response by tokenizing the text alone.
                text_only = self.tokenizer(
                    text, add_special_tokens=True, return_tensors="pt", truncation=False
                )["input_ids"].squeeze(0)
                text_end_position = len(text_only)

        cls_position = max(0, text_end_position - 1)

        if full_length > self.max_length:
            full_input_ids = full_input_ids[: self.max_length]
            text_end_position = min(text_end_position, self.max_length)
            cls_position = min(cls_position, self.max_length - 1)
            actual_length = self.max_length
        else:
            actual_length = full_length

        pad_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )
        padding = self.max_length - len(full_input_ids)
        if padding > 0:
            full_input_ids = torch.cat(
                [full_input_ids, torch.full((padding,), pad_id, dtype=full_input_ids.dtype)]
            )

        attention_mask = torch.zeros(self.max_length, dtype=torch.long)
        attention_mask[:actual_length] = 1

        labels = full_input_ids.clone()
        labels[:text_end_position] = -100
        labels[attention_mask == 0] = -100

        return {
            "input_ids": full_input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "severity_labels": torch.tensor(level, dtype=torch.long),
            "cls_position": torch.tensor(cls_position, dtype=torch.long),
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "severity_labels": torch.stack([b["severity_labels"] for b in batch]),
        "cls_position": torch.stack([b["cls_position"] for b in batch]),
    }


class SkipDataLoader:
    """Wrap a DataLoader to skip an initial number of batches on resume."""

    def __init__(self, dataloader: DataLoader, skip_steps: int = 0) -> None:
        self.dataloader = dataloader
        self.skip_steps = skip_steps

    def __iter__(self):
        iterator = iter(self.dataloader)
        for _ in range(self.skip_steps):
            try:
                next(iterator)
            except StopIteration:
                return
        yield from iterator

    def __len__(self) -> int:
        return max(0, len(self.dataloader) - self.skip_steps)


# =============================================================================
# Config
# =============================================================================

@dataclass
class TrainConfig:
    # Mode (set from YAML 'mode' section)
    target_format: str = "full_sequence"
    use_classification_head: bool = True
    use_classification_only: bool = False  # implies use_generation_head=False

    # Model
    backbone: str = "mistralai/Ministral-8B-Instruct-2410"
    quantization: str = "4bit"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    compute_dtype: str = "bfloat16"
    attn_implementation: Optional[str] = "flash_attention_2"

    # LoRA
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: str = "all_linear"  # or list of module names

    # Classifier head
    cls_hidden_dim: int = 512
    cls_dropout: float = 0.1
    num_classes: int = 5

    # Loss
    alpha_cls: float = 0.5
    use_weighted_cross_entropy: bool = True
    inverse_frequency_gamma: float = 1.0
    use_ordinal_penalty: bool = True
    ordinal_lambda: float = 0.5
    loss_combination: str = "simple_sum"  # 'simple_sum' or 'uncertainty_weighting'

    # Data
    train_data: str = ""
    val_data: Optional[str] = None
    val_split_ratio: float = 0.01
    max_seq_length: int = 2048
    max_samples: Optional[int] = None

    # Training
    output_dir: str = "runs/stage1"
    num_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate_backbone: float = 2.0e-4
    learning_rate_classifier: float = 2.0e-3
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    seed: int = 42
    save_interval_ratio: float = 0.1

    resume_from_checkpoint: Optional[str] = None

    # ------------------------------------------------------------------ Loaders

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        cfg = cls()
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        def set_block(block: Dict[str, Any], remap: Dict[str, str]) -> None:
            for src_key, dst_attr in remap.items():
                if src_key in block:
                    setattr(cfg, dst_attr, block[src_key])

        mode = data.get("mode", {})
        set_block(
            mode,
            {
                "target_format": "target_format",
                "use_classification_head": "use_classification_head",
                "use_classification_only": "use_classification_only",
            },
        )

        model_section = data.get("model", {})
        set_block(
            model_section,
            {
                "backbone": "backbone",
                "quantization": "quantization",
                "bnb_4bit_quant_type": "bnb_4bit_quant_type",
                "bnb_4bit_use_double_quant": "bnb_4bit_use_double_quant",
                "compute_dtype": "compute_dtype",
                "attn_implementation": "attn_implementation",
            },
        )

        lora = data.get("lora", {})
        set_block(
            lora,
            {
                "r": "lora_r",
                "alpha": "lora_alpha",
                "dropout": "lora_dropout",
                "target_modules": "lora_target_modules",
            },
        )

        head = data.get("classifier_head", {})
        set_block(
            head,
            {
                "hidden_dim": "cls_hidden_dim",
                "dropout": "cls_dropout",
                "num_classes": "num_classes",
            },
        )

        loss = data.get("loss", {})
        set_block(
            loss,
            {
                "alpha_cls": "alpha_cls",
                "use_weighted_cross_entropy": "use_weighted_cross_entropy",
                "inverse_frequency_gamma": "inverse_frequency_gamma",
                "use_ordinal_penalty": "use_ordinal_penalty",
                "ordinal_lambda": "ordinal_lambda",
                "loss_combination": "loss_combination",
            },
        )

        data_section = data.get("data", {})
        set_block(
            data_section,
            {
                "max_seq_length": "max_seq_length",
                "val_split_ratio": "val_split_ratio",
            },
        )

        training = data.get("training", {})
        set_block(
            training,
            {
                "num_epochs": "num_epochs",
                "per_device_train_batch_size": "per_device_train_batch_size",
                "gradient_accumulation_steps": "gradient_accumulation_steps",
                "learning_rate": "learning_rate_backbone",
                "learning_rate_backbone": "learning_rate_backbone",
                "learning_rate_classifier": "learning_rate_classifier",
                "warmup_ratio": "warmup_ratio",
                "lr_scheduler_type": "lr_scheduler_type",
                "weight_decay": "weight_decay",
                "max_grad_norm": "max_grad_norm",
                "seed": "seed",
                "save_interval_ratio": "save_interval_ratio",
            },
        )
        return cfg

    def apply_cli_overrides(self, args: argparse.Namespace) -> None:
        for key, value in vars(args).items():
            if value is None or key in {"config"}:
                continue
            if hasattr(self, key):
                setattr(self, key, value)

    def derived(self) -> Dict[str, Any]:
        use_generation_head = not self.use_classification_only
        if self.use_classification_only:
            mode = "classification_only"
        elif use_generation_head and self.use_classification_head:
            mode = "dual_head"
        elif use_generation_head and not self.use_classification_head:
            mode = (
                "triage_full_context"
                if self.target_format == "full_sequence"
                else "triage_level"
            )
        else:
            mode = "unknown"
        return {
            "use_generation_head": use_generation_head,
            "use_classification_head": self.use_classification_head
            or self.use_classification_only,
            "mode": mode,
        }


# =============================================================================
# Build model
# =============================================================================

def _torch_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def build_model(cfg: TrainConfig, device: torch.device) -> Tuple[KTASUnifiedModel, Any]:
    compute_dtype = _torch_dtype(cfg.compute_dtype)
    if cfg.quantization == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
        )
    elif cfg.quantization in (None, "none", "fp16", "bf16"):
        bnb_config = None
    else:
        raise ValueError(f"Unsupported quantization={cfg.quantization!r}")

    backbone_kwargs: Dict[str, Any] = {
        "quantization_config": bnb_config,
        "torch_dtype": compute_dtype,
    }
    if cfg.attn_implementation:
        backbone_kwargs["attn_implementation"] = cfg.attn_implementation

    base_model = AutoModelForCausalLM.from_pretrained(cfg.backbone, **backbone_kwargs)
    base_model.config.use_cache = False

    if cfg.quantization == "4bit":
        base_model = prepare_model_for_kbit_training(
            base_model, use_gradient_checkpointing=True
        )

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg.lora_target_modules,
    )
    base_model = get_peft_model(base_model, lora_config)

    derived = cfg.derived()
    hidden_size = base_model.config.hidden_size

    model = KTASUnifiedModel(
        base_model=base_model,
        hidden_size=hidden_size,
        num_classes=cfg.num_classes,
        cls_hidden_dim=cfg.cls_hidden_dim,
        cls_dropout=cfg.cls_dropout,
        cls_dtype=compute_dtype,
        use_generation_head=derived["use_generation_head"],
        use_classification_head=derived["use_classification_head"],
    ).to(device)
    return model, lora_config


# =============================================================================
# Loss / optimizer construction
# =============================================================================

def build_classification_loss(
    cfg: TrainConfig, train_levels: List[int], device: torch.device
) -> Optional[WeightedOrdinalCrossEntropy]:
    derived = cfg.derived()
    if not derived["use_classification_head"]:
        return None
    weights = None
    if cfg.use_weighted_cross_entropy:
        weights = compute_inverse_frequency_weights(
            train_levels, num_classes=cfg.num_classes, gamma=cfg.inverse_frequency_gamma
        ).tolist()
    return WeightedOrdinalCrossEntropy(
        num_classes=cfg.num_classes,
        class_weights=weights,
        lambda_ord=cfg.ordinal_lambda if cfg.use_ordinal_penalty else 0.0,
    ).to(device)


def build_optimizer_and_scheduler(
    cfg: TrainConfig,
    model: KTASUnifiedModel,
    uncertainty_weighting: Optional[UncertaintyWeighting],
    num_training_steps: int,
) -> Tuple[torch.optim.Optimizer, Any]:
    derived = cfg.derived()
    param_groups: List[Dict[str, Any]] = [
        {
            "params": [
                p for _, p in model.base_model.named_parameters() if p.requires_grad
            ],
            "lr": cfg.learning_rate_backbone,
        }
    ]
    if derived["use_classification_head"] and model.severity_classifier is not None:
        param_groups.append(
            {
                "params": list(model.severity_classifier.parameters()),
                "lr": cfg.learning_rate_classifier,
            }
        )
    if uncertainty_weighting is not None:
        param_groups.append(
            {
                "params": list(uncertainty_weighting.parameters()),
                "lr": cfg.learning_rate_classifier,
            }
        )

    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)
    warmup_steps = max(1, int(num_training_steps * cfg.warmup_ratio))
    if cfg.lr_scheduler_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, warmup_steps, num_training_steps
        )
    elif cfg.lr_scheduler_type == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer, warmup_steps, num_training_steps
        )
    else:
        raise ValueError(f"Unsupported lr_scheduler_type={cfg.lr_scheduler_type!r}")
    return optimizer, scheduler


# =============================================================================
# Train / eval
# =============================================================================

def train_one_epoch(
    *,
    cfg: TrainConfig,
    model: KTASUnifiedModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    cls_loss_fn: Optional[WeightedOrdinalCrossEntropy],
    loss_combiner: Optional[LossCombiner],
    epoch: int,
    global_step: int,
    save_interval: int,
    save_fn: Optional[Any],
) -> Tuple[Dict[str, float], int]:
    model.train()
    derived = cfg.derived()
    use_cls = derived["use_classification_head"]
    use_gen = derived["use_generation_head"]

    accum = cfg.gradient_accumulation_steps
    totals: Dict[str, float] = {"loss": 0.0, "gen_loss": 0.0, "cls_loss": 0.0, "n": 0.0}
    micro_step = 0
    iterator = (
        tqdm(loader, desc=f"Epoch {epoch}", disable=not is_main_process()) if save_fn else loader
    )

    for batch in iterator:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"] if use_gen else None,
            cls_positions=batch["cls_position"] if use_cls else None,
        )

        gen_loss = outputs.get("gen_loss") if use_gen else None
        cls_loss = None
        if use_cls and cls_loss_fn is not None:
            cls_loss = cls_loss_fn(outputs["severity_logits"], batch["severity_labels"])

        if use_gen and use_cls and loss_combiner is not None:
            combined, _info = loss_combiner.combine(gen_loss, cls_loss)
        elif use_gen and use_cls:
            combined = gen_loss + cfg.alpha_cls * cls_loss
        elif use_gen:
            combined = gen_loss
        else:
            combined = cls_loss

        loss = combined / accum
        loss.backward()

        totals["loss"] += float(combined.item())
        totals["n"] += 1.0
        if gen_loss is not None:
            totals["gen_loss"] += float(gen_loss.item())
        if cls_loss is not None:
            totals["cls_loss"] += float(cls_loss.item())

        micro_step += 1
        if micro_step % accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        if save_fn is not None and save_interval > 0 and micro_step % save_interval == 0:
            save_fn(epoch=epoch, micro_step=micro_step, global_step=global_step, metrics=totals)

    # Flush remaining gradients
    if micro_step % accum != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

    n = max(totals["n"], 1.0)
    return (
        {
            "loss": totals["loss"] / n,
            "gen_loss": totals["gen_loss"] / n,
            "cls_loss": totals["cls_loss"] / n,
        },
        global_step,
    )


@torch.no_grad()
def evaluate(
    *,
    cfg: TrainConfig,
    model: KTASUnifiedModel,
    loader: DataLoader,
    device: torch.device,
    cls_loss_fn: Optional[WeightedOrdinalCrossEntropy],
) -> Dict[str, float]:
    model.eval()
    derived = cfg.derived()
    use_cls = derived["use_classification_head"]
    use_gen = derived["use_generation_head"]

    totals = {"gen_loss": 0.0, "cls_loss": 0.0, "cls_correct": 0.0, "n": 0.0, "n_cls": 0.0}
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"] if use_gen else None,
            cls_positions=batch["cls_position"] if use_cls else None,
        )
        if use_gen and outputs.get("gen_loss") is not None:
            totals["gen_loss"] += float(outputs["gen_loss"].item())
        if use_cls and cls_loss_fn is not None:
            cls_loss = cls_loss_fn(outputs["severity_logits"], batch["severity_labels"])
            totals["cls_loss"] += float(cls_loss.item())
            preds = outputs["severity_logits"].argmax(dim=-1)
            totals["cls_correct"] += float((preds == batch["severity_labels"]).sum().item())
            totals["n_cls"] += float(batch["severity_labels"].numel())
        totals["n"] += 1.0

    n = max(totals["n"], 1.0)
    n_cls = max(totals["n_cls"], 1.0)
    return {
        "val_gen_loss": totals["gen_loss"] / n,
        "val_cls_loss": totals["cls_loss"] / n,
        "val_cls_accuracy": totals["cls_correct"] / n_cls,
    }


# =============================================================================
# Checkpointing
# =============================================================================

def save_checkpoint(
    *,
    path: Path,
    cfg: TrainConfig,
    model: KTASUnifiedModel,
    tokenizer,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    uncertainty_weighting: Optional[UncertaintyWeighting],
    epoch: int,
    micro_step: int,
    global_step: int,
    total_steps_per_epoch: int,
    metrics: Dict[str, Any],
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    derived = cfg.derived()

    model.base_model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    if derived["use_classification_head"] and model.severity_classifier is not None:
        torch.save(model.severity_classifier.state_dict(), path / "severity_classifier.pt")
    if uncertainty_weighting is not None:
        torch.save(uncertainty_weighting.state_dict(), path / "uncertainty_weighting.pt")

    torch.save(
        {
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "global_step": global_step,
        },
        path / "training_state.pt",
    )
    (path / "checkpoint_info.json").write_text(
        json.dumps(
            {
                "epoch": epoch,
                "step": micro_step,
                "global_step": global_step,
                "total_steps_per_epoch": total_steps_per_epoch,
                "metrics": metrics,
                "mode": derived["mode"],
                "saved_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            },
            indent=2,
        )
    )


def maybe_resume(
    *,
    path: Optional[str],
    model: KTASUnifiedModel,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    uncertainty_weighting: Optional[UncertaintyWeighting],
    device: torch.device,
    total_steps_per_epoch: int,
) -> Tuple[int, int, int]:
    """Return ``(start_epoch, start_step, global_step)``.

    Loads optimizer/scheduler/cls-head/uncertainty state from ``path`` if
    present; otherwise starts fresh from epoch 1.
    """
    if not path:
        return 1, 0, 0

    base = Path(path)
    cls_path = base / "severity_classifier.pt"
    uw_path = base / "uncertainty_weighting.pt"
    state_path = base / "training_state.pt"
    info_path = base / "checkpoint_info.json"

    if cls_path.exists() and model.severity_classifier is not None:
        model.severity_classifier.load_state_dict(torch.load(cls_path, map_location=device))
    if uw_path.exists() and uncertainty_weighting is not None:
        uncertainty_weighting.load_state_dict(torch.load(uw_path, map_location=device))

    if not state_path.exists():
        if is_main_process():
            logger.warning("training_state.pt missing in %s; starting fresh.", path)
        return 1, 0, 0

    state = torch.load(state_path, map_location=device)
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])
    global_step = int(state.get("global_step", 0))

    start_epoch, start_step = 1, 0
    if info_path.exists():
        info = json.loads(info_path.read_text())
        saved_epoch = int(info.get("epoch", 1))
        saved_step = int(info.get("step", 0))
        saved_total = int(info.get("total_steps_per_epoch", total_steps_per_epoch))
        if saved_step >= saved_total:
            start_epoch = saved_epoch + 1
            start_step = 0
        else:
            start_epoch = saved_epoch
            start_step = saved_step

    if is_main_process():
        logger.info(
            "Resumed from %s at epoch=%d step=%d (global_step=%d)",
            path,
            start_epoch,
            start_step,
            global_step,
        )
    return start_epoch, start_step, global_step


# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1 trainer for Dr.KTAS")
    parser.add_argument("--config", type=str, required=True, help="Path to a YAML config")
    parser.add_argument("--train_data", type=str, default=None)
    parser.add_argument("--val_data", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--learning_rate_backbone", type=float, default=None)
    parser.add_argument("--learning_rate_classifier", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig.from_yaml(args.config)
    cfg.apply_cli_overrides(args)

    if not cfg.train_data:
        raise SystemExit(
            "train_data is required. Pass --train_data or set it in the YAML config."
        )

    # DDP setup
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    derived = cfg.derived()
    if is_main_process():
        logger.info("Mode: %s", derived["mode"])
        logger.info("Target format: %s", cfg.target_format)
        logger.info("Backbone: %s", cfg.backbone)
        logger.info("Quantization: %s", cfg.quantization)
        logger.info(
            "Effective batch = %d (per device) x %d (accum) x %d (GPUs) = %d",
            cfg.per_device_train_batch_size,
            cfg.gradient_accumulation_steps,
            world_size,
            cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps * world_size,
        )

    # Tokenizer and dataset
    tokenizer = setup_tokenizer(cfg.backbone)

    dataset = KTASDataset(
        data_path=cfg.train_data,
        tokenizer=tokenizer,
        max_length=cfg.max_seq_length,
        target_format=cfg.target_format,
        max_samples=cfg.max_samples,
    )
    if cfg.val_data:
        val_dataset = KTASDataset(
            data_path=cfg.val_data,
            tokenizer=tokenizer,
            max_length=cfg.max_seq_length,
            target_format=cfg.target_format,
        )
        train_dataset = dataset
    else:
        n_val = max(1, int(len(dataset) * cfg.val_split_ratio))
        n_train = len(dataset) - n_val
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(cfg.seed),
        )

    train_sampler = (
        DistributedSampler(train_dataset, shuffle=True) if world_size > 1 else None
    )
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if world_size > 1 else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.per_device_train_batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=world_size > 1,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.per_device_train_batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )

    # Model
    model, _ = build_model(cfg, device)
    if is_main_process():
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(
            "Trainable parameters: %s / %s (%.2f%%)",
            f"{trainable:,}",
            f"{total:,}",
            100 * trainable / max(total, 1),
        )

    # Loss
    train_levels = []
    base_dataset = (
        train_dataset.dataset if isinstance(train_dataset, torch.utils.data.Subset) else train_dataset
    )
    indices = (
        train_dataset.indices if isinstance(train_dataset, torch.utils.data.Subset) else range(len(base_dataset))
    )
    for idx in indices:
        train_levels.append(base_dataset.samples[idx]["level"] + 1)

    cls_loss_fn = build_classification_loss(cfg, train_levels, device)

    uncertainty_weighting: Optional[UncertaintyWeighting] = None
    loss_combiner: Optional[LossCombiner] = None
    if derived["use_classification_head"] and derived["use_generation_head"]:
        if cfg.loss_combination == "uncertainty_weighting":
            uncertainty_weighting = UncertaintyWeighting(num_tasks=2).to(device)
            loss_combiner = LossCombiner(
                method="uncertainty_weighting",
                uncertainty_weighting=uncertainty_weighting,
            )
        else:
            loss_combiner = LossCombiner(
                method="simple_sum", cls_weight=cfg.alpha_cls
            )

    # Optimizer
    total_steps_per_epoch = len(train_loader)
    num_training_steps = max(
        1, (total_steps_per_epoch // cfg.gradient_accumulation_steps) * cfg.num_epochs
    )
    optimizer, scheduler = build_optimizer_and_scheduler(
        cfg, model, uncertainty_weighting, num_training_steps
    )

    start_epoch, start_step, global_step = maybe_resume(
        path=cfg.resume_from_checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        uncertainty_weighting=uncertainty_weighting,
        device=device,
        total_steps_per_epoch=total_steps_per_epoch,
    )

    save_interval = max(1, int(total_steps_per_epoch * cfg.save_interval_ratio))

    def save_latest(*, epoch: int, micro_step: int, global_step: int, metrics: Dict[str, Any]) -> None:
        if not is_main_process():
            return
        save_checkpoint(
            path=output_dir / "latest_model",
            cfg=cfg,
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            uncertainty_weighting=uncertainty_weighting,
            epoch=epoch,
            micro_step=micro_step,
            global_step=global_step,
            total_steps_per_epoch=total_steps_per_epoch,
            metrics=metrics,
        )

    # Training loop
    for epoch in range(start_epoch, cfg.num_epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        effective_loader: Any = train_loader
        if epoch == start_epoch and start_step > 0:
            effective_loader = SkipDataLoader(train_loader, skip_steps=start_step)

        train_metrics, global_step = train_one_epoch(
            cfg=cfg,
            model=model,
            loader=effective_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            cls_loss_fn=cls_loss_fn,
            loss_combiner=loss_combiner,
            epoch=epoch,
            global_step=global_step,
            save_interval=save_interval,
            save_fn=save_latest,
        )

        val_metrics = evaluate(
            cfg=cfg, model=model, loader=val_loader, device=device, cls_loss_fn=cls_loss_fn
        )

        if is_main_process():
            logger.info(
                "Epoch %d done. train=%s val=%s",
                epoch,
                {k: round(v, 4) for k, v in train_metrics.items()},
                {k: round(v, 4) for k, v in val_metrics.items()},
            )
            save_latest(
                epoch=epoch,
                micro_step=total_steps_per_epoch,
                global_step=global_step,
                metrics={**train_metrics, **val_metrics},
            )

    if is_main_process():
        final_path = output_dir / "final_adapter"
        save_checkpoint(
            path=final_path,
            cfg=cfg,
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            uncertainty_weighting=uncertainty_weighting,
            epoch=cfg.num_epochs,
            micro_step=total_steps_per_epoch,
            global_step=global_step,
            total_steps_per_epoch=total_steps_per_epoch,
            metrics={"status": "completed"},
        )
        logger.info("Training complete. Final adapter at %s", final_path)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
