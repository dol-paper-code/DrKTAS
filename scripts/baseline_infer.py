#!/usr/bin/env python3
"""Zero-shot baseline inference for the paper.

Supports three backbones through a single ``--model`` flag:

* ``ministral8b``  — ``mistralai/Ministral-8B-Instruct-2410`` (HF, 4-bit QLoRA optional)
* ``meditron14b``  — ``OpenMeditron/Meditron3-Qwen2.5-14B``    (HF, 4-bit QLoRA optional)
* ``gpt4o``        — ``gpt-4o-2024-08-06``                      (OpenAI Batch API or single)

For Hugging Face backends, ``KTASFirstTokenConstrainedProcessor`` is used by
default to restrict the first generated token to one of the digits ``1`` to
``5`` (or the EOS token afterwards), matching the constrained-decoding
treatment in the paper.

The script writes a single JSON file with one entry per test sample. The
companion script ``baseline_evaluate.py`` consumes this file and reports
the agreement, ordinal, and safety-oriented metrics defined in
``src/drktas/metrics.py``.

Examples
--------

    # Ministral-8B zero-shot, 1 GPU
    python scripts/baseline_infer.py \
        --model ministral8b --test_data path/to/your_test.csv \
        --output runs/baseline_ministral8b.json

    # Meditron-14B zero-shot, 4 GPUs
    torchrun --nproc_per_node=4 scripts/baseline_infer.py \
        --model meditron14b --test_data path/to/your_test.csv \
        --output runs/baseline_meditron14b.json

    # GPT-4o zero-shot via Batch API (set OPENAI_API_KEY first)
    OPENAI_API_KEY=sk-... python scripts/baseline_infer.py \
        --model gpt4o --test_data path/to/your_test.csv \
        --use_batch_api --output runs/baseline_gpt4o.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import re
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Allow running from the repository root without installation.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from drktas.data_io import extract_level_from_fullseverity  # noqa: E402
from drktas.prompts import load_prompt  # noqa: E402


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("drktas.baseline_infer")


# =============================================================================
# Backbone registry
# =============================================================================

@dataclass(frozen=True)
class BackboneSpec:
    name: str
    hf_model_id: Optional[str] = None
    openai_model_id: Optional[str] = None
    family: str = "hf"  # 'hf' or 'openai'


BACKBONES: Dict[str, BackboneSpec] = {
    "ministral8b": BackboneSpec(
        name="ministral8b",
        hf_model_id="mistralai/Ministral-8B-Instruct-2410",
        family="hf",
    ),
    "meditron14b": BackboneSpec(
        name="meditron14b",
        hf_model_id="OpenMeditron/Meditron3-Qwen2.5-14B",
        family="hf",
    ),
    "gpt4o": BackboneSpec(
        name="gpt4o",
        openai_model_id="gpt-4o-2024-08-06",
        family="openai",
    ),
}


# =============================================================================
# KTAS grade info & prompts
# =============================================================================

KTAS_GRADE_INFO = load_prompt("ktas_grade_info")


def extract_patient_info(text: str) -> str:
    """Strip leading sections so the prompt focuses on the patient record."""
    info = text
    if "[환자 기록]" in text:
        info = text[text.find("[환자 기록]"):]
    return info.replace("[중증도]", "").strip()


def build_prompt(text: str, prompt_type: str) -> str:
    if prompt_type not in {"no_description", "with_description", "constrained"}:
        raise ValueError(f"Unknown prompt_type={prompt_type!r}.")
    template = load_prompt(f"baseline_{prompt_type}")
    if prompt_type == "no_description":
        # The test record already contains the patient information and the
        # KTAS question; the template is just a pass-through formatter.
        return template.format(patient_info=text)
    info = extract_patient_info(text)
    return template.format(patient_info=info, grade_info=KTAS_GRADE_INFO)


# =============================================================================
# Response parsing
# =============================================================================

_BRACKET_PATTERN = re.compile(r"\[\[([1-5])\]\]")
_LABELED_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r"KTAS\s*등급\s*[:\s]\s*([1-5])", re.IGNORECASE),
    re.compile(r"등급\s*[:\s]\s*([1-5])", re.IGNORECASE),
    re.compile(r"Grade\s*[:\s]\s*([1-5])", re.IGNORECASE),
)
_LEAD_NUM = re.compile(r"^\s*([1-5])\s*")
_TAIL_NUM = re.compile(r"([1-5])\s*$")
_BARE_DIGIT = re.compile(r"(?<![0-9])([1-5])(?![0-9])")


def parse_prediction(prediction: str, strict: bool = False) -> Optional[int]:
    """Recover a KTAS level (1-5) from a free-text model response.

    Search order:
      1. Exact single digit
      2. ``[[N]]`` bracket pattern
      3. Labeled patterns (``KTAS 등급: N`` / ``Grade: N``)
      4. Leading or trailing single digit (skipped when ``strict``)
      5. Last bare digit in the cleaned response (skipped when ``strict``)

    Returns ``None`` when no candidate can be extracted.
    """
    if not prediction:
        return None
    s = prediction.strip()
    if s in ("1", "2", "3", "4", "5"):
        return int(s)

    m = _BRACKET_PATTERN.search(s)
    if m:
        return int(m.group(1))
    for pat in _LABELED_PATTERNS:
        m = pat.search(s)
        if m:
            return int(m.group(1))

    if strict:
        return None

    head = _LEAD_NUM.match(s[:10])
    if head:
        return int(head.group(1))
    tail = _TAIL_NUM.search(s)
    if tail:
        return int(tail.group(1))

    cleaned = re.sub(r"Grade\s+[1-5]\s*\([^)]+\)", "", s, flags=re.IGNORECASE)
    cleaned = re.sub(r"\[KTAS[^\]]*\]", "", cleaned)
    digits = _BARE_DIGIT.findall(cleaned)
    if digits:
        return int(digits[-1])
    return None


# =============================================================================
# Test data
# =============================================================================

def load_test_data(path: str | Path, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load a CSV of test records into a list of dicts.

    Required columns: ``text``, ``fullseverity``. Optional identifier
    columns are preserved on the output for join-back.
    """
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_samples is not None and i >= max_samples:
                break
            text = row.get("text", "").strip("\r").lstrip()
            fullseverity = row.get("fullseverity", "").strip()
            level = extract_level_from_fullseverity(fullseverity)
            if level is None:
                continue
            rows.append(
                {
                    "index": i,
                    "text": text,
                    "fullseverity": fullseverity,
                    "true_label": level,
                    "Research_ID": row.get("Research_ID"),
                    "Extracted_ID": row.get("Extracted_ID"),
                }
            )
    logger.info("Loaded %d samples from %s", len(rows), path)
    return rows


# =============================================================================
# Backend abstraction
# =============================================================================

class Backend(ABC):
    """Common interface across HF and OpenAI backends."""

    name: str

    @abstractmethod
    def generate(self, prompts: List[str]) -> List[str]:
        """Return one response per prompt."""


# ----------------------------- HF backend -----------------------------------

class HFBackend(Backend):
    """Hugging Face Transformers backend with optional constrained decoding."""

    def __init__(
        self,
        spec: BackboneSpec,
        *,
        max_new_tokens: int = 50,
        use_4bit: bool = True,
        use_constrained_decoding: bool = True,
        local_rank: int = 0,
        world_size: int = 1,
    ) -> None:
        import torch
        from transformers import (  # noqa: F401
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            LogitsProcessor,
            LogitsProcessorList,
        )

        if spec.hf_model_id is None:
            raise ValueError(f"Backbone {spec.name} has no HF model id.")
        self.spec = spec
        self.name = spec.name
        self.max_new_tokens = max_new_tokens
        self.use_constrained_decoding = use_constrained_decoding
        self.local_rank = local_rank
        self.world_size = world_size
        self._torch = torch
        self._LogitsProcessor = LogitsProcessor
        self._LogitsProcessorList = LogitsProcessorList

        logger.info("Loading %s (constrained=%s)", spec.hf_model_id, use_constrained_decoding)

        self.tokenizer = AutoTokenizer.from_pretrained(spec.hf_model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        bnb_config = None
        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )

        device_map = {"": local_rank} if world_size > 1 else "auto"
        self.model = AutoModelForCausalLM.from_pretrained(
            spec.hf_model_id,
            quantization_config=bnb_config,
            torch_dtype=compute_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device

        self.logits_processor = (
            self._build_constrained_processor() if use_constrained_decoding else None
        )

    # ------------------------------------------------------------------ Helpers

    def _build_constrained_processor(self):
        """First-token KTAS-grade constrained logits processor."""
        torch = self._torch
        tokenizer = self.tokenizer

        def _find_grade_tokens() -> set:
            ids: set = set()
            for grade in (1, 2, 3, 4, 5):
                for text in (
                    str(grade),
                    f" {grade}",
                    f"{grade}\n",
                    f"**{grade}**",
                    f"[[{grade}]]",
                ):
                    try:
                        encoded = tokenizer.encode(text, add_special_tokens=False)
                        if encoded:
                            ids.add(encoded[0])
                    except Exception:
                        continue
            return ids

        allowed = _find_grade_tokens()
        eos_id = tokenizer.eos_token_id

        class _FirstTokenProcessor(self._LogitsProcessor):
            def __init__(self) -> None:
                super().__init__()
                self._is_first = True

            def __call__(self, input_ids, scores):
                mask = torch.full_like(scores, float("-inf"))
                if self._is_first:
                    for tid in allowed:
                        if tid < scores.shape[-1]:
                            mask[:, tid] = scores[:, tid]
                    self._is_first = False
                else:
                    if eos_id is not None:
                        mask[:, eos_id] = 0.0
                return mask

        return _FirstTokenProcessor

    # ------------------------------------------------------------------ Generate

    def generate(self, prompts: List[str]) -> List[str]:
        torch = self._torch
        if not prompts:
            return []
        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048
        ).to(self.device)

        processors = None
        if self.logits_processor is not None:
            processors = self._LogitsProcessorList([self.logits_processor()])

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                logits_processor=processors,
            )

        input_len = inputs["input_ids"].shape[1]
        return [
            self.tokenizer.decode(out[input_len:], skip_special_tokens=True).strip()
            for out in outputs
        ]


# --------------------------- OpenAI backend ---------------------------------

class OpenAIBackend(Backend):
    """OpenAI backend with optional Batch API."""

    def __init__(
        self,
        spec: BackboneSpec,
        *,
        max_new_tokens: int = 50,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: Optional[int] = 42,
        max_retries: int = 5,
        backoff_min_s: float = 2.0,
        backoff_max_s: float = 60.0,
        use_batch_api: bool = False,
        batch_output_dir: Optional[Path] = None,
    ) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit(
                "OPENAI_API_KEY environment variable is required for the GPT-4o baseline."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise SystemExit(
                "The 'openai' package is required for the GPT-4o baseline. "
                "Install it via `pip install openai`."
            ) from exc

        if spec.openai_model_id is None:
            raise ValueError(f"Backbone {spec.name} has no OpenAI model id.")
        self.spec = spec
        self.name = spec.name
        self.client = OpenAI(api_key=api_key)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.max_retries = max_retries
        self.backoff_min_s = backoff_min_s
        self.backoff_max_s = backoff_max_s
        self.use_batch_api = use_batch_api
        self.batch_output_dir = (
            Path(batch_output_dir) if batch_output_dir is not None else None
        )

    # ------------------------------------------------------------------ Single

    def _call_chat_completion(self, prompt: str) -> str:
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                payload: Dict[str, Any] = {
                    "model": self.spec.openai_model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "n": 1,
                    "max_completion_tokens": self.max_new_tokens,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                }
                if self.seed is not None:
                    payload["seed"] = self.seed
                completion = self.client.chat.completions.create(**payload)
                return (completion.choices[0].message.content or "").strip()
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
                if status == 400:
                    raise RuntimeError(f"OpenAI request invalid (400): {exc}") from exc
                sleep_s = min(
                    self.backoff_max_s,
                    self.backoff_min_s * (2 ** attempt) + random.random(),
                )
                time.sleep(sleep_s)
        raise RuntimeError(f"OpenAI API failed after retries: {last_err}") from last_err

    # ------------------------------------------------------------------ Batch

    def _run_batch_api(self, prompts: List[str]) -> List[str]:
        if self.batch_output_dir is None:
            self.batch_output_dir = Path("openai_batch_logs")
        self.batch_output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        request_path = self.batch_output_dir / f"batch_requests_{timestamp}.jsonl"
        with request_path.open("w", encoding="utf-8") as f:
            for i, prompt in enumerate(prompts):
                request = {
                    "custom_id": f"request-{i}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self.spec.openai_model_id,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_completion_tokens": self.max_new_tokens,
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        **({"seed": self.seed} if self.seed is not None else {}),
                    },
                }
                f.write(json.dumps(request, ensure_ascii=False) + "\n")
        logger.info("Wrote %d batch requests to %s", len(prompts), request_path)

        with request_path.open("rb") as f:
            uploaded = self.client.files.create(file=f, purpose="batch")
        batch = self.client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": "drktas baseline"},
        )
        batch_id = batch.id
        logger.info("Submitted OpenAI batch: %s", batch_id)

        # Poll until the batch terminates.
        terminal = {"completed", "failed", "cancelled", "expired"}
        while True:
            status = self.client.batches.retrieve(batch_id)
            if status.status in terminal:
                break
            time.sleep(30)
            logger.info("Batch %s status: %s", batch_id, status.status)

        if status.status != "completed":
            raise RuntimeError(f"OpenAI batch {batch_id} ended with status {status.status}")

        output_path = self.batch_output_dir / f"batch_results_{timestamp}.jsonl"
        file_content = self.client.files.content(status.output_file_id).read()
        output_path.write_bytes(file_content)

        # Parse outputs back into prompt order.
        outputs_by_id: Dict[str, str] = {}
        for raw in file_content.decode("utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            record = json.loads(raw)
            custom_id = record.get("custom_id")
            try:
                outputs_by_id[custom_id] = (
                    record["response"]["body"]["choices"][0]["message"]["content"] or ""
                ).strip()
            except (KeyError, TypeError):
                outputs_by_id[custom_id] = ""
        return [outputs_by_id.get(f"request-{i}", "") for i in range(len(prompts))]

    # ------------------------------------------------------------------ Generate

    def generate(self, prompts: List[str]) -> List[str]:
        if self.use_batch_api:
            return self._run_batch_api(prompts)
        return [self._call_chat_completion(p) for p in prompts]


# =============================================================================
# Driver
# =============================================================================

def make_backend(model: str, args: argparse.Namespace) -> Backend:
    spec = BACKBONES[model]
    if spec.family == "openai":
        return OpenAIBackend(
            spec,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
            use_batch_api=args.use_batch_api,
            batch_output_dir=Path(args.openai_batch_dir) if args.openai_batch_dir else None,
        )
    if spec.family == "hf":
        return HFBackend(
            spec,
            max_new_tokens=args.max_new_tokens,
            use_4bit=not args.no_4bit,
            use_constrained_decoding=not args.no_constrained_decoding,
            local_rank=args.local_rank if args.local_rank >= 0 else 0,
            world_size=max(1, int(os.environ.get("WORLD_SIZE", 1))),
        )
    raise ValueError(f"Unknown backbone family {spec.family!r}.")


def _setup_distributed_hf() -> Tuple[int, int]:
    """Initialize ``torch.distributed`` for HF backends when launched via torchrun."""
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if world_size > 1:
        import torch
        import torch.distributed as dist

        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    return local_rank, world_size


def run_inference(
    backend: Backend,
    samples: List[Dict[str, Any]],
    prompt_type: str,
    batch_size: int,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    n = len(samples)
    for start in range(0, n, batch_size):
        chunk = samples[start : start + batch_size]
        prompts = [build_prompt(sample["text"], prompt_type) for sample in chunk]
        responses = backend.generate(prompts)
        for sample, raw in zip(chunk, responses):
            pred = parse_prediction(raw)
            results.append(
                {
                    "index": sample["index"],
                    "true_label": sample["true_label"],
                    "prediction": pred,
                    "parse_failed": pred is None,
                    "raw_response": raw,
                    "Research_ID": sample.get("Research_ID"),
                    "Extracted_ID": sample.get("Extracted_ID"),
                }
            )
        if (start // batch_size + 1) % 10 == 0 or start + batch_size >= n:
            done = min(start + batch_size, n)
            ok = sum(1 for r in results if not r["parse_failed"])
            logger.info(
                "Progress: %d / %d (parse OK: %d / %d, %.1f%%)",
                done,
                n,
                ok,
                done,
                100.0 * ok / max(done, 1),
            )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zero-shot baseline inference")
    parser.add_argument("--model", choices=sorted(BACKBONES.keys()), required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument(
        "--prompt_type",
        choices=("no_description", "with_description", "constrained"),
        default="with_description",
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1)

    # HF-specific
    parser.add_argument("--no_4bit", action="store_true")
    parser.add_argument("--no_constrained_decoding", action="store_true")

    # OpenAI-specific
    parser.add_argument("--use_batch_api", action="store_true")
    parser.add_argument(
        "--openai_batch_dir",
        type=str,
        default=None,
        help="Directory where batch request / result JSONL files are written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if BACKBONES[args.model].family == "hf":
        local_rank, world_size = _setup_distributed_hf()
        args.local_rank = local_rank
    else:
        world_size = 1

    samples = load_test_data(args.test_data, args.max_samples)

    backend = make_backend(args.model, args)
    results = run_inference(backend, samples, args.prompt_type, args.batch_size)

    if world_size > 1:
        # Each rank writes its own shard; downstream evaluation supports
        # multi-file globs through the ``--predictions`` argument.
        output_path = Path(args.output)
        output_path = output_path.with_name(
            f"{output_path.stem}_rank{args.local_rank}{output_path.suffix}"
        )
    else:
        output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    document = {
        "model": args.model,
        "spec": asdict(BACKBONES[args.model]),
        "prompt_type": args.prompt_type,
        "config": {
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
            "use_4bit": not args.no_4bit,
            "use_constrained_decoding": not args.no_constrained_decoding,
            "use_batch_api": args.use_batch_api,
            "seed": args.seed,
            "test_data": args.test_data,
            "world_size": world_size,
        },
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "results": results,
    }
    output_path.write_text(json.dumps(document, indent=2, ensure_ascii=False))
    logger.info("Wrote %d results to %s", len(results), output_path)

    if world_size > 1:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
