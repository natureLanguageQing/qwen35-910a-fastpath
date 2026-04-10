# Qwen3.5-9B on Ascend 910A Fastpath Notes

This repository fragment documents a practical fastpath configuration for running Qwen3.5-9B-class models on Ascend 910A. It focuses on decode hot paths observed in recurrent/state-update workloads and separates what has been measured from what is recommended for multi-card deployment.

No model weights, private hostnames, credentials, internal absolute paths, or service endpoints are included.

## Scope

The optimization target is low-level decode performance, not HTTP proxy tricks. The important hot paths are:

- `fused_step`: small decode step with recurrent matmul plus state update.
- `batch_step`: batched decode step, especially `duplicate_rounds_fastpath` and `unique_fastpath`.
- `state_update`: the recurrent state transition path.

## Latest Low-Level Patch

Added patch script:

- `scripts/patches/patch_recurrent_decode_fused_step_skip_redundant_kv.py`

What it changes:

- In `qwen35_gdn_recurrent_decode_packed.cpp` fused-step device path, when the packed `state@[k,q]` fastpath already produced `kv_mem`, skip the second redundant `state@k` matmul.
- This keeps backend routing consistent and removes duplicate compute in the same decode step.

The current best strategy is a mixed route:

- Use `state_update_registered_stub` for `batch_step`.
- Use shape-aware `fp16_qkv_only -> state_add_opcommand` buckets for selected small `fused_step` shapes.
- Keep `ascendc_state_update_host_stub` for small shapes where it wins.
- Do not route large `batch_step` through the AscendC host-stub fallback path.
- Avoid the direct monolithic TBE `StateUpdate` path until its schedule/lowering is fixed.

## Fastest Single-Card Scheme

Use one Ascend 910A card with the recurrent decode extension enabled. This is the best-validated configuration so far.

Recommended route:

- `batch_step duplicate` -> `duplicate_rounds_fastpath + state_update_registered_stub`
- `batch_step unique` -> `unique_fastpath + state_update_registered_stub`
- `fused_step` selected buckets -> `state_add_opcommand`
- other small `fused_step` shapes -> `ascendc_state_update_host_stub`

Recommended bucket list:

```text
128x128,128x96,96x128,96x96,96x160,160x96
```

Use the environment template in [configs/single_card_fast.env](configs/single_card_fast.env).

Practical performance statement for external sharing:

- Single-card single-stream: about `10 tok/s`
- Single-card aggregate throughput: about `206 tok/s @ concurrency=64`

These two numbers are serving-facing statements. They should not be mixed with the much higher recurrent microbench throughput numbers shown later in this document.

Measured microbench samples:

| Case | Current fastpath | Fallback | Result |
|---|---:|---:|---:|
| batch=48, states=96, duplicate, 128x128 | 10223 tok/s | 8027 tok/s | current wins |
| batch=48, states=96, unique, 128x128 | 14821 tok/s | 8897 tok/s | current wins |
| batch=64, states=96, duplicate, 128x128 | 12181 tok/s | 8699 tok/s | current wins |
| batch=64, states=96, unique, 128x128 | 16242 tok/s | 8577 tok/s | current wins |
| batch=64, states=128, duplicate, 96x96 | 12529 tok/s | 8791 tok/s | current wins |
| batch=64, states=128, unique, 96x96 | 15986 tok/s | 10535 tok/s | current wins |

For default small-shape probing, the completed partial result also favored the current route:

```text
batch=32, states=64, unique, 128x128
current:  unique_fastpath + state_update_registered_stub, 12931 tok/s
fallback: addcmul_state_update, 8716 tok/s
```

## Fastest Dual-Card Scheme

For two cards, the recommended fastest practical scheme is two independent single-card workers behind a load balancer, not tensor parallelism by default.

Why:

- Qwen3.5-9B-class fp16 inference can fit on a single 32 GB-class 910A card in many deployments, depending on runtime and KV/cache pressure.
- The measured low-level wins are intra-card decode-path wins.
- Tensor parallelism can add HCCL/all-reduce overhead and may reduce single-stream latency unless the model or context length forces sharding.
- Independent workers are usually better for aggregate throughput and service availability.

Recommended topology:

```text
client / router
  -> worker-0 on NPU 0, single-card fast config
  -> worker-1 on NPU 1, single-card fast config
```

Use [configs/dual_card_fast.env](configs/dual_card_fast.env) as a template and start two workers with different `ASCEND_RT_VISIBLE_DEVICES` and ports.

If the target is strict single-stream latency rather than aggregate throughput, benchmark both:

- two independent workers, one request pinned to one card
- framework-native TP=2, if your runtime supports stable Qwen3.5-9B on 910A

Do not assume TP=2 is faster without measuring.

## Fastest Multi-Card Scheme

For more than two cards, the recommended open-source baseline is a pool of independent single-card workers. Scale the number of replicas first; introduce TP/PP only when memory, context length, or prefill load requires it.

Recommended topology:

```text
router
  -> text worker pool: N independent single-card workers
  -> vision/multimodal worker pool: separate workers, lower concurrency, larger timeout
```

Why separate pools:

- Text decode and image prefill have different bottlenecks.
- Image requests can block or inflate latency for text-only traffic.
- The low-level decode fastpath should remain stable for text and small decode chunks.

Use [configs/multi_card_fast.env](configs/multi_card_fast.env) as a starting point.

## What Not To Enable By Default

Avoid these as default fast paths:

- Monolithic direct TBE `Qwen35GdnRecurrentDecodePackedStateUpdate`; it reached TBE compile/lowering but failed in the schedule/lowering path in the tested stack.
- Routing large `batch_step` through the AscendC host-stub fallback path; it can collapse large-BH throughput.
- Forcing `192x192` into the `fp16_qkv_only` bucket; sampled results showed it was slower than the host-stub route.
- Blindly enabling tensor parallelism for a 9B model that already fits on one card.

## Reproducibility Notes

Benchmarks in this package are microbench results, not a universal serving SLA. Always re-run on your own stack because 910A behavior depends on:

- CANN version
- `torch_npu` version
- custom OPP installation
- HCCL/network topology
- model context length
- request mix between text and multimodal

The benchmark source summary is in [benchmarks/results_summary.md](benchmarks/results_summary.md).
