#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


TARGETS = [
    Path("/root/src/ascend_ops_stage/recurrent_decode_packed_mvp/python/csrc/qwen35_gdn_recurrent_decode_packed.cpp"),
    Path("/root/qwen910a_rebuild/qwen35_gdn_recurrent_decode_packed.cpp"),
]


OLD = "  if (use_fused_step_direct_aclnn()) {\n    if (supports_generic_aclnn_batch_matmul(\n"
NEW = "  if (!kv_mem.defined() && use_fused_step_direct_aclnn()) {\n    if (supports_generic_aclnn_batch_matmul(\n"


def patch_one(path: Path) -> None:
    if not path.exists():
        print("skip missing", path)
        return
    text = path.read_text(encoding="utf-8")
    if NEW in text:
        print("already patched", path)
        return
    if OLD not in text:
        raise RuntimeError(f"pattern not found in {path}")
    text = text.replace(OLD, NEW, 1)
    path.write_text(text, encoding="utf-8")
    print("patched", path)


def main() -> None:
    for target in TARGETS:
        patch_one(target)


if __name__ == "__main__":
    main()
