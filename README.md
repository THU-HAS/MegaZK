# MegaZK (open-source)

GPU-accelerated Plonky2 proving.

This repository is the open-source implementation of **MegaZK**, the system
described in the ICS '26 paper
[MegaZK: A Memory Efficient GPU System Accelerating End-to-end Zero-Knowledge Proof](https://dl.acm.org/doi/10.1145/3797905.3807842)
(DOI: [10.1145/3797905.3807842](https://doi.org/10.1145/3797905.3807842)).
Citation: Muyang Li, Yueteng Yu, Bangyan Wang, Xiong Fan, Mingyu Gao, and
Shuwen Deng. 2026. In *Proceedings of the 40th ACM International Conference on
Supercomputing* (ICS '26). ACM, 1165–1176.
https://doi.org/10.1145/3797905.3807842

The GPU proving path lives in the forked [`plonky2`](plonky2) crate and calls
into [`era-boojum-cuda`](era-boojum-cuda) / [`era-cuda`](era-cuda) for CUDA
kernels and runtime bindings.

The default Cargo feature set is CPU-only and does not build CUDA dependencies.
In the benchmark programs, `--baseline` selects the CPU prover; the GPU path is
opt-in with `--features gpu`. Running a non-baseline benchmark without that
feature returns `GPU proving is unavailable; rebuild with --features gpu`.

This monorepo is dual-licensed **MIT OR Apache-2.0**. See [`LICENSE`](LICENSE)
and [`NOTICE`](NOTICE).

## Repository Layout

- [`plonky2`](plonky2): forked Plonky2 workspace. GPU proving enters through
  `build_gpu` / `GpuCircuitData::prove` and the `plonky2/plonky2/src/integration`
  module.
- [`era-boojum-cuda`](era-boojum-cuda): CUDA kernels and Rust wrappers used by
  the GPU prover.
- [`era-cuda`](era-cuda): CUDA runtime bindings.
- [`plonky2-sha256`](plonky2-sha256), [`plonky2-sha512`](plonky2-sha512),
  [`plonky2-examples`](plonky2-examples), [`proto-neural-zkp`](proto-neural-zkp):
  benchmark applications.
- [`plonky2_ed25519`](plonky2_ed25519): experimental Ed25519 benchmark. Proof
  generation is expected to run, but verification currently has a known issue
  (`BENCHMARK_VERIFY_STATUS: KNOWN_ISSUE`). That is not a regression.
- [`plonky2-gpu-bench`](plonky2-gpu-bench): std-only internal helper for shared
  benchmark timing, verification policy, and CUDA memory reporting.

## Current Reproduction Environment

The following environment is the current known reproduction baseline. Other
versions may work, but have not been normalized for this open-source snapshot.

- OS: Ubuntu 22.04.5 LTS, Linux 6.8.0-124-generic, x86_64
- Rust: `nightly-2023-11-28` (pinned in [`rust-toolchain.toml`](rust-toolchain.toml))
  - `rustc 1.76.0-nightly (49b3924bd 2023-11-27)`
  - `cargo 1.76.0-nightly (9b13310ca 2023-11-24)`
- CUDA toolkit: 12.x (this snapshot was built with 12.5, `nvcc V12.5.82`)
- CMake: 3.30.3
- clang: 14.0.0
- GPU/driver: requires an NVIDIA GPU and driver compatible with CUDA 12.x. If
  CMake cannot detect the GPU architecture during build, set `CUDAARCHS`
  manually, for example `CUDAARCHS=89`.

The repository has a pinned toolchain in [`rust-toolchain.toml`](rust-toolchain.toml).
For CPU baseline comparisons, `RUSTFLAGS=-Ctarget-cpu=native` can improve
performance on the local machine.

## Support Matrix

| Benchmark | GPU proof | CPU baseline | Verification status |
| --- | --- | --- | --- |
| Factorial | Supported | Supported | Expected to pass |
| Fibonacci | Supported | Supported | Expected to pass |
| SHA256 | Supported | Supported | Expected to pass |
| SHA512 | Supported | Supported | Expected to pass |
| ECDSA | Supported | Supported | Expected to pass |
| MVM | Supported | Supported | Expected to pass |
| Ed25519 | Experimental | Experimental | Known verification issue |

Ed25519: prove is expected to run; verify is a known failure (`KNOWN_ISSUE`),
not a regression.

## Benchmark Circuit Sources

These are the **open-source circuit / gadget implementations** used by each
benchmark (not the proving-system libraries). Commands below are this
repository's CLI; full smoke and paper-scale invocations are in
[Build And Run](#build-and-run). Paper numbers in brackets are ICS '26
camera-ready references. No pinned commit is recorded in this snapshot, so
links go to the upstream repository root.

- **Factorial** — run: `-p plonky2 --example factorial` (for example
  `-- --num 2000`). Circuit:
  [0xPolygonZero/plonky2](https://github.com/0xPolygonZero/plonky2)
  (paper [27]; in-tree
  [`plonky2/plonky2/examples/factorial.rs`](plonky2/plonky2/examples/factorial.rs)).
- **Fibonacci** — run: `-p plonky2 --example fibonacci` (for example
  `-- --num 2000`). Circuit:
  [0xPolygonZero/plonky2](https://github.com/0xPolygonZero/plonky2)
  (paper [27]; in-tree
  [`plonky2/plonky2/examples/fibonacci.rs`](plonky2/plonky2/examples/fibonacci.rs)).
- **SHA256** — run: `-p plonky2_sha256` (for example `-- --num 1`). Circuit:
  [polymerdao/plonky2-sha256](https://github.com/polymerdao/plonky2-sha256)
  (paper [30]; in-tree [`plonky2-sha256/`](plonky2-sha256)).
- **SHA512** — run: `-p plonky2_sha512` (for example `-- --num 1`). Circuit:
  [polymerdao/plonky2-sha512](https://github.com/polymerdao/plonky2-sha512)
  (paper [31]; in-tree [`plonky2-sha512/`](plonky2-sha512)).
- **ECDSA** — run: `-p plonky2-examples --example 4_ecdsa_verification`
  (for example `-- --num 1`). Circuit example:
  [qope/plonky2-examples](https://github.com/qope/plonky2-examples)
  (paper [36]; in-tree [`plonky2-examples/`](plonky2-examples)). ECDSA gadget
  used by that example:
  [0xPolygonZero/plonky2-ecdsa](https://github.com/0xPolygonZero/plonky2-ecdsa)
  (in-tree [`plonky2-ecdsa/`](plonky2-ecdsa)).
- **Ed25519** — run: `-p plonky2_ed25519` (for example `-- --num 1`). Circuit
  vendored in this tree:
  [Electron-Labs/plonky2_ed25519](https://github.com/Electron-Labs/plonky2_ed25519)
  (in-tree [`plonky2_ed25519/`](plonky2_ed25519)). Paper [29] cites
  [polymerdao/plonky2-ed25519](https://github.com/polymerdao/plonky2-ed25519);
  the files here match Electron-Labs (authors, `curve25519-dalek` git rev, and
  layout), not the polymerdao tree.
- **MVM** — run: `-p neural-zkp --bin rust-app` (for example
  `-- --input-size 100 --output-size 100`). Circuit:
  [worldcoin/proto-neural-zkp](https://github.com/worldcoin/proto-neural-zkp)
  (paper [43]; in-tree [`proto-neural-zkp/`](proto-neural-zkp)). The vendored
  `Cargo.toml` still lists
  `https://github.com/dcbuild3r/proto-neural-zkp/` as `repository`; that is a
  contributor copy of the Worldcoin project.

## Build And Run

Run the following commands from the **repository root**.

The default feature set is CPU-only. Use `--features gpu` for the GPU prover,
and `--baseline` for the CPU prover in the benchmark CLIs.

This is a root Cargo workspace. Commands below use `--locked` so they build from
the checked-in root `Cargo.lock`. rustup will pick `nightly-2023-11-28` from
[`rust-toolchain.toml`](rust-toolchain.toml). GPU builds need CUDA 12.x, an
NVIDIA driver, and (when CMake cannot detect the architecture) `CUDAARCHS`.

The commands in this README are **this repository's CLI**.

Every benchmark reports `Time taken to build:`, `Time taken to prove:`, and
`Time taken to verify:` using Rust `Duration` formatting. GPU runs also report
used and free CUDA memory with a shared format. MVM retains its existing
seven-column CSV output in addition to these human-readable timings. A completed
verification also emits exactly one `BENCHMARK_VERIFY_STATUS: PASS`,
`BENCHMARK_VERIFY_STATUS: KNOWN_ISSUE`, or `BENCHMARK_VERIFY_STATUS: FAIL`
machine-readable status line.

### Smoke (small workloads)

#### Factorial

```bash
cargo run --locked --release -p plonky2 --features gpu --example factorial -- --num 2000
cargo run --locked --release -p plonky2 --example factorial -- --num 2000 --baseline
```

#### Fibonacci

```bash
cargo run --locked --release -p plonky2 --features gpu --example fibonacci -- --num 2000
cargo run --locked --release -p plonky2 --example fibonacci -- --num 2000 --baseline
```

#### SHA256

```bash
cargo run --locked --release -p plonky2_sha256 --features gpu -- --num 1
cargo run --locked --release -p plonky2_sha256 -- --num 1 --baseline
```

#### SHA512

```bash
cargo run --locked --release -p plonky2_sha512 --features gpu -- --num 1
cargo run --locked --release -p plonky2_sha512 -- --num 1 --baseline
```

#### ECDSA

```bash
cargo run --locked --release -p plonky2-examples --features gpu --example 4_ecdsa_verification -- --num 1
cargo run --locked --release -p plonky2-examples --example 4_ecdsa_verification -- --num 1 --baseline
```

#### Ed25519 Experimental

```bash
cargo run --locked --release -p plonky2_ed25519 --features gpu -- --num 1
cargo run --locked --release -p plonky2_ed25519 -- --num 1 --baseline
```

Ed25519 proof generation is expected to run. A verification failure after proof
generation is a known issue (`BENCHMARK_VERIFY_STATUS: KNOWN_ISSUE`) and should
not be treated as a regression.

#### MVM

```bash
cargo run --locked --release -p neural-zkp --features gpu --bin rust-app -- --input-size 100 --output-size 100
cargo run --locked --release -p neural-zkp --bin rust-app -- --input-size 100 --output-size 100 --baseline
```

### Paper-scale examples

`--memory-profile` defaults to `compressed`. Paper-scale runs need large GPU
memory (about 24GB class). Factorial and Fibonacci paper-scale **sorting is
very slow**; prove can still run.

Uncompressed (paper-U size flags):

```bash
cargo run --locked --release -p plonky2 --features gpu --example factorial -- --num 131073 --memory-profile uncompressed
cargo run --locked --release -p plonky2 --features gpu --example fibonacci -- --num 262143 --memory-profile uncompressed
cargo run --locked --release -p plonky2_sha256 --features gpu -- --num 60 --memory-profile uncompressed
cargo run --locked --release -p plonky2_sha512 --features gpu -- --num 20 --memory-profile uncompressed
cargo run --locked --release -p plonky2-examples --features gpu --example 4_ecdsa_verification -- --num 6 --seed 5eed5eed5eed5eed --memory-profile uncompressed
cargo run --locked --release -p plonky2_ed25519 --features gpu -- --num 2 --memory-profile uncompressed
cargo run --locked --release -p neural-zkp --features gpu --bin rust-app -- --input-size 4000 --output-size 4000 --coefficient-bits 16 --num-wires 400 --num-routed-wires 400 --seed 5eed5eed5eed5eed --memory-profile uncompressed
```

Compressed (paper-C size flags; default profile is already `compressed`):

```bash
cargo run --locked --release -p plonky2 --features gpu --example factorial -- --num 524289 --memory-profile compressed
cargo run --locked --release -p plonky2 --features gpu --example fibonacci -- --num 1048575 --memory-profile compressed
cargo run --locked --release -p plonky2_sha256 --features gpu -- --num 210 --memory-profile compressed
cargo run --locked --release -p plonky2_sha512 --features gpu -- --num 137 --memory-profile compressed
cargo run --locked --release -p plonky2-examples --features gpu --example 4_ecdsa_verification -- --num 42 --seed 5eed5eed5eed5eed --memory-profile compressed
cargo run --locked --release -p plonky2_ed25519 --features gpu -- --num 11 --memory-profile compressed
cargo run --locked --release -p neural-zkp --features gpu --bin rust-app -- --input-size 9000 --output-size 9000 --coefficient-bits 16 --num-wires 400 --num-routed-wires 400 --seed 5eed5eed5eed5eed --memory-profile compressed
```

Ed25519 paper-scale: prove is expected to run; verify remains the same known
failure (`KNOWN_ISSUE`).

## Notes For Contributors

- This repository is a root Cargo workspace. Use `--locked` for reproducible
  builds against the checked-in root `Cargo.lock`.
- The `plonky2` crate and benchmark packages keep CUDA support behind their
  non-default `gpu` feature. Targeted default builds are CPU-only.
- The GPU path uses `CircuitBuilder::build_gpu` with witness input targets,
  followed by `GpuCircuitData::prove` with an equally sized witness input slice.
  GPU proofs continue to use the CPU verifier through `GpuCircuitData::verify`.
- Verbose proving features may produce JSON debug dumps. These files are local
  artifacts and should not be committed.

## Acknowledgments

Proving-system libraries vendored in this tree (not the benchmark circuits
above). Details are in [`NOTICE`](NOTICE):

- **plonky2**: [0xPolygonZero/plonky2](https://github.com/0xPolygonZero/plonky2)
  (also [mir-protocol/plonky2](https://github.com/mir-protocol/plonky2)); this
  copy lives in [`plonky2/`](plonky2)
- **era-boojum**: [matter-labs/era-boojum](https://github.com/matter-labs/era-boojum)
  ([`era-boojum/`](era-boojum))
- **era-cuda**: [matter-labs/era-cuda](https://github.com/matter-labs/era-cuda)
  ([`era-cuda/`](era-cuda))
- **era-boojum-cuda**: [matter-labs/era-boojum-cuda](https://github.com/matter-labs/era-boojum-cuda)
  ([`era-boojum-cuda/`](era-boojum-cuda))
