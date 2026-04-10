#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


TARGET_BASENAME = "qwen35_gdn_recurrent_decode_packed.cpp"


OLD = "  if (use_fused_step_direct_aclnn()) {\n    if (supports_generic_aclnn_batch_matmul(\n"
NEW = "  if (!kv_mem.defined() && use_fused_step_direct_aclnn()) {\n    if (supports_generic_aclnn_batch_matmul(\n"


def discover_targets(search_root: Path) -> list[Path]:
    return sorted(
        {
            path.resolve()
            for path in search_root.rglob(TARGET_BASENAME)
            if path.is_file()
        }
    )


def patch_one(path: Path) -> str:
    if not path.exists():
        print("skip missing", path)
        return "missing"
    text = path.read_text(encoding="utf-8")
    if NEW in text:
        print("already patched", path)
        return "already"
    if OLD not in text:
        print("pattern not found", path)
        return "pattern_not_found"
    text = text.replace(OLD, NEW, 1)
    path.write_text(text, encoding="utf-8")
    print("patched", path)
    return "patched"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Patch qwen35 recurrent decode fused-step path to skip redundant kv "
            "matmul when kv_mem is already defined."
        )
    )
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        help="Explicit target file path. Can be repeated.",
    )
    parser.add_argument(
        "--search-root",
        default=".",
        help=(
            "Root directory used for auto-discovery when --target is not provided. "
            "Default: current directory."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.target:
        targets = [Path(item).expanduser() for item in args.target]
    else:
        search_root = Path(args.search_root).expanduser()
        targets = discover_targets(search_root)
        if not targets:
            print(
                (
                    "error: no target discovered. pass --target /path/to/"
                    f"{TARGET_BASENAME} or set --search-root."
                ),
                file=sys.stderr,
            )
            return 2
        print(f"discovered {len(targets)} target(s) under {search_root.resolve()}:")
        for target in targets:
            print(" -", target)

    patched = 0
    already = 0
    missing = 0
    pattern_not_found = 0

    for target in targets:
        status = patch_one(target)
        if status == "patched":
            patched += 1
        elif status == "already":
            already += 1
        elif status == "missing":
            missing += 1
        elif status == "pattern_not_found":
            pattern_not_found += 1

    if pattern_not_found:
        print(
            f"error: pattern not found in {pattern_not_found} target(s); patch not applied.",
            file=sys.stderr,
        )
        return 3

    if not (patched or already):
        print(
            "error: no target patched and no target already patched.",
            file=sys.stderr,
        )
        if missing:
            print(f"detail: {missing} target(s) missing.", file=sys.stderr)
        return 4

    print(
        f"summary: patched={patched}, already_patched={already}, missing={missing}, "
        f"pattern_not_found={pattern_not_found}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
