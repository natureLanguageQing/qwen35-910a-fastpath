#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from typing import Callable


def build_counter(tokenizer_dir: str | None) -> tuple[str, Callable[[str], int]]:
    if not tokenizer_dir:
        return "chars", lambda text: len(text)
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "Failed to import transformers. Install it or omit --tokenizer-dir."
        ) from exc
    tok = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    return "tokens", lambda text: len(tok.encode(text, add_special_tokens=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible base URL")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--prompt", default="请直接回答，不要展示思考过程。用中文简短介绍你自己。")
    parser.add_argument(
        "--tokenizer-dir",
        default=None,
        help="Optional tokenizer path for true token counting. If unset, uses char counting.",
    )
    parser.add_argument(
        "--ignore-empty-output",
        action="store_true",
        help="Ignore runs with 0 output units when aggregating average throughput.",
    )
    args = parser.parse_args()

    unit_name, counter = build_counter(args.tokenizer_dir)
    vals: list[float] = []
    for i in range(args.runs):
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": args.prompt}],
            "stream": False,
            "max_tokens": args.max_tokens,
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            args.base_url.rstrip("/") + "/v1/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=300) as resp:
            obj = json.loads(resp.read().decode("utf-8"))
        dt = time.perf_counter() - t0
        text = obj["choices"][0]["message"]["content"]
        units = counter(text)
        units_s = units / dt if dt > 0 else 0.0
        keep = not (args.ignore_empty_output and units == 0)
        if keep:
            vals.append(units_s)
        print(
            f"run{i + 1}: {dt:.3f}s, {unit_name}={units}, "
            f"{unit_name}/s={units_s:.3f}, kept={int(keep)}"
        )

    if not vals:
        raise RuntimeError("No valid runs left after filtering.")
    print(f"avg_{unit_name}/s={sum(vals) / len(vals):.3f}")


if __name__ == "__main__":
    main()
