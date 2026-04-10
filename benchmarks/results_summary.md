# Benchmark Summary

These numbers are microbenchmarks from a 910A optimization pass. They are intended to document route decisions, not to promise universal serving throughput.

## Batch Step

Boundary probe results:

| batch | states | pattern | K | V | current backend | current tok/s | fallback backend | fallback tok/s | winner |
|---:|---:|---|---:|---:|---|---:|---|---:|---|
| 48 | 96 | duplicate | 128 | 128 | state_update_registered_stub | 10223.58 | addcmul_state_update | 8027.30 | current |
| 48 | 96 | unique | 128 | 128 | state_update_registered_stub | 14821.80 | addcmul_state_update | 8897.95 | current |
| 64 | 96 | duplicate | 128 | 128 | state_update_registered_stub | 12181.31 | addcmul_state_update | 8699.87 | current |
| 64 | 96 | unique | 128 | 128 | state_update_registered_stub | 16242.53 | addcmul_state_update | 8577.46 | current |
| 64 | 128 | duplicate | 96 | 96 | state_update_registered_stub | 12529.26 | addcmul_state_update | 8791.90 | current |
| 64 | 128 | unique | 96 | 96 | state_update_registered_stub | 15986.21 | addcmul_state_update | 10535.17 | current |

Conclusion:

```text
batch_step should keep state_update_registered_stub.
Do not route large batch_step through the AscendC host-stub fallback path.
```

## Fused Step

Recommended default bucket list:

```text
128x128,128x96,96x128,96x96,96x160,160x96
```

Observed routing rule:

- Use `fp16_qkv_only -> state_add_opcommand` for the bucket list above.
- Keep `ascendc_state_update_host_stub` for shapes where sampled forcing was slower, such as `192x192`.

Representative sampled values:

| HV | K | V | preferred route | helper_fused_step |
|---:|---:|---:|---|---:|
| 16 | 128 | 128 | state_add_opcommand | about 1.05 ms |
| 32 | 128 | 128 | state_add_opcommand | about 2.04 ms |
| 32 | 128 | 96 | state_add_opcommand | about 1.56 ms |
| 32 | 96 | 128 | state_add_opcommand | about 3.02 ms |
| 32 | 96 | 96 | state_add_opcommand | about 2.27 ms |
| 16 | 192 | 192 | ascendc_state_update_host_stub | about 2.27 ms |

## Known Bad / Risky Paths

- Direct monolithic TBE `StateUpdate` reached compilation/lowering but failed on the tested stack.
- `StateApply` TBE variants also hit vector/lowering issues in the tested stack.
- Split-device-op through `DeltaState + StateAdd` can be much faster than CPU fallback for small state-update probes, but the route must be shape-aware.
- Large `batch_step` must be protected from accidentally falling into the wrong small-shape route.

