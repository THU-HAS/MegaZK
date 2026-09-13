## plonky2 Ed25519

Contains a Plonky2 implementation of the [Ed25519 signature scheme](https://datatracker.ietf.org/doc/html/rfc8032#section-6).

Proof generation is expected to run. Verification currently fails; that is a
known issue (`BENCHMARK_VERIFY_STATUS: KNOWN_ISSUE`), not a regression.

GPU and CPU-baseline commands (repository root, `--locked`, `--features gpu`)
are documented in the root [README](../README.md). Do not use CPU-only upstream
`cargo run` invocations as the GPU path.

Smoke (from the repository root):

```bash
cargo run --locked --release -p plonky2_ed25519 --features gpu -- --num 1
cargo run --locked --release -p plonky2_ed25519 -- --num 1 --baseline
```
