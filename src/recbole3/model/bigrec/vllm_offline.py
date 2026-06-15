"""Offline vLLM beam-search generation subprocess for BIGRec.

Launched by :class:`recbole3.model.bigrec.trainer.BIGRecTrainer` as a one-shot
subprocess (in the user-configured ``vllm_conda_env``) during evaluation.
The subprocess reads prompts from a JSON file, runs ``LLM.beam_search`` with
:class:`vllm.sampling_params.BeamSearchParams`, then writes the top-1
generated text per prompt to an output JSON file.

This replaces the previous OpenAI-compatible HTTP server path because vLLM
removed ``use_beam_search`` from the SamplingParams / completions endpoint in
0.6.4.  ``LLM.beam_search`` is the only supported way to get true width-N
beam search from vLLM ≥ 0.6.4 (including 0.19.x).

**Standalone by design.**  This file is invoked by absolute path
(not via ``python -m``) so the vLLM conda environment only needs ``vllm``
installed — it does NOT need to ``import recbole3``.  Only ``argparse``,
``json``, ``sys`` (stdlib) and ``vllm`` are imported here.

Invocation contract (all arguments required unless marked optional)::

    /path/to/vllm_env/python /path/to/vllm_offline.py \\
        --prompts INPUT.json --output OUTPUT.json \\
        --model BASE_MODEL_PATH \\
        --beam-width 4 --max-tokens 128 --max-model-len 576 \\
        [--lora LORA_DIR --lora-rank 8] \\
        [--dtype float16] \\
        [--tp 1] \\
        [--gpu-memory-utilization 0.9]

Input JSON: ``list[str]`` of full prompts (already including ``### Response:\\n``).
Output JSON: ``list[str]`` of generated texts (top-1 beam, prompt stripped).
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the offline subprocess."""
    parser = argparse.ArgumentParser(
        description="Offline vLLM beam-search generation for BIGRec evaluation.",
    )
    parser.add_argument("--prompts", required=True, help="Input JSON file: list of prompt strings.")
    parser.add_argument("--output", required=True, help="Output JSON file: list of generated texts.")
    parser.add_argument("--model", required=True, help="Base model path or HF hub identifier.")
    parser.add_argument("--lora", default="", help="LoRA adapter directory (empty: no LoRA).")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank (sets max_lora_rank).")
    parser.add_argument("--dtype", default="float16", help="Model dtype: float16 or bfloat16.")
    parser.add_argument("--beam-width", type=int, required=True, help="Beam width (num_beams).")
    parser.add_argument("--max-tokens", type=int, required=True, help="Max new tokens per generation.")
    parser.add_argument("--max-model-len", type=int, required=True, help="Max model context length.")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size.")
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=0.9,
        help="Fraction of GPU VRAM vLLM pre-allocates.",
    )
    return parser.parse_args()


def _extract_top1_text(beam_output: Any) -> str:
    """Extract the top-1 generated text from a single ``LLM.beam_search`` result.

    The exact attribute layout of vLLM's ``BeamSearchOutput`` has shifted across
    versions; this helper checks the two known shapes and falls back to empty
    string so the caller can detect missing generations.

    Args:
        beam_output: One element of the list returned by ``LLM.beam_search``.

    Returns:
        Generated text for the highest-log-probability beam.
    """
    # vLLM ≥ 0.6.4: BeamSearchOutput has .sequences = list[BeamSearchSequence],
    # sorted by descending cumulative log-probability.  Each sequence has a
    # `.text` attribute holding the decoded continuation.
    sequences = getattr(beam_output, "sequences", None)
    if sequences:
        text = getattr(sequences[0], "text", None)
        if text is not None:
            return text

    # Fallback: some vLLM versions expose `.outputs` (mirroring RequestOutput).
    outputs = getattr(beam_output, "outputs", None)
    if outputs:
        text = getattr(outputs[0], "text", None)
        if text is not None:
            return text

    return ""


def main() -> int:
    """Entry point: load model, run beam search, write results."""
    args = _parse_args()

    # Heavy import deferred so --help works without vLLM installed.
    from vllm import LLM
    from vllm.sampling_params import BeamSearchParams

    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "dtype": args.dtype,
        "tensor_parallel_size": args.tp,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
    }
    if args.lora:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = args.lora_rank

    print(f"[vllm_offline] Loading model: {args.model}", flush=True)
    llm = LLM(**llm_kwargs)

    with open(args.prompts, "r", encoding="utf-8") as f:
        prompts: list[str] = json.load(f)
    print(f"[vllm_offline] Loaded {len(prompts)} prompts.", flush=True)

    params = BeamSearchParams(
        beam_width=args.beam_width,
        max_tokens=args.max_tokens,
        temperature=0.0,
    )

    lora_request: Any = None
    if args.lora:
        from vllm.lora.request import LoRARequest
        # Positional args: (lora_name, lora_int_id, lora_path).  Field-name
        # variants have changed across vLLM versions; positional is stable.
        lora_request = LoRARequest("bigrec_adapter", 1, args.lora)

    print(
        f"[vllm_offline] Running beam_search (width={args.beam_width}, "
        f"max_tokens={args.max_tokens}, lora={'on' if args.lora else 'off'}) …",
        flush=True,
    )
    if lora_request is not None:
        outputs = llm.beam_search(prompts, params, lora_request=lora_request)
    else:
        outputs = llm.beam_search(prompts, params)

    top1_texts: list[str] = [_extract_top1_text(out) for out in outputs]
    empty_count = sum(1 for t in top1_texts if not t)
    if empty_count:
        print(
            f"[vllm_offline] WARNING: {empty_count}/{len(top1_texts)} generations "
            "returned empty text; check vLLM BeamSearchOutput layout for this version.",
            flush=True,
        )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(top1_texts, f, ensure_ascii=False)
    print(f"[vllm_offline] Wrote {len(top1_texts)} generations to {args.output}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
