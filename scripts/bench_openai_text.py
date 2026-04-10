#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import urllib.request


def count_tokens_rough(text: str) -> int:
    # Keep this script dependency-light for open-source smoke testing.
    return max(1, len(text))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible base URL")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--prompt", default="请直接回答，不要展示思考过程。用中文简短介绍你自己。")
    args = parser.parse_args()

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
        rough_tokens = count_tokens_rough(text)
        char_s = rough_tokens / dt if dt > 0 else 0.0
        vals.append(char_s)
        print(f"run{i + 1}: {dt:.3f}s, rough_chars={rough_tokens}, rough_chars/s={char_s:.3f}")

    print(f"avg_rough_chars/s={sum(vals) / len(vals):.3f}")


if __name__ == "__main__":
    main()

