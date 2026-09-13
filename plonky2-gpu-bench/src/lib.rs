use std::fmt::Display;
use std::str::FromStr;
use std::time::{Duration, Instant};

pub const VERIFY_STATUS_PASS: &str = "BENCHMARK_VERIFY_STATUS: PASS";
pub const VERIFY_STATUS_KNOWN_ISSUE: &str = "BENCHMARK_VERIFY_STATUS: KNOWN_ISSUE";
pub const VERIFY_STATUS_FAIL: &str = "BENCHMARK_VERIFY_STATUS: FAIL";
pub const KNOWN_ISSUE_WARNING_SENTINEL: &str =
    "WARNING: benchmark known issue: verification failed:";

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum MemoryProfile {
    #[default]
    Compressed,
    Uncompressed,
}

impl MemoryProfile {
    pub fn lde_counts(self, uncompressed: [usize; 4]) -> [usize; 4] {
        match self {
            Self::Compressed => [0; 4],
            Self::Uncompressed => uncompressed,
        }
    }
}

impl Display for MemoryProfile {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Compressed => formatter.write_str("compressed"),
            Self::Uncompressed => formatter.write_str("uncompressed"),
        }
    }
}

impl FromStr for MemoryProfile {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "compressed" => Ok(Self::Compressed),
            "uncompressed" => Ok(Self::Uncompressed),
            _ => Err(format!(
                "invalid memory profile {value:?}; expected compressed or uncompressed"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VerifyPolicy {
    Strict,
    KnownIssue { message: &'static str },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum VerifyOutcome {
    Verified,
    KnownIssue,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PhaseTimings {
    pub build: Duration,
    pub prove: Duration,
    pub verify: Duration,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RunReport {
    pub timings: PhaseTimings,
}

#[derive(Debug)]
pub struct BenchSession {
    timings: PhaseTimings,
}

impl BenchSession {
    pub fn new() -> Self {
        Self {
            timings: PhaseTimings::default(),
        }
    }

    pub fn build<T>(&mut self, operation: impl FnOnce() -> T) -> T {
        let started = Instant::now();
        let value = operation();
        self.timings.build = started.elapsed();
        println!("Time taken to build: {:?}", self.timings.build);
        value
    }

    pub fn prove<T>(&mut self, operation: impl FnOnce() -> T) -> T {
        let started = Instant::now();
        let value = operation();
        self.timings.prove = started.elapsed();
        println!("Time taken to prove: {:?}", self.timings.prove);
        value
    }

    pub fn verify<E: Display>(
        &mut self,
        policy: VerifyPolicy,
        operation: impl FnOnce() -> Result<(), E>,
    ) -> Result<VerifyOutcome, E> {
        let started = Instant::now();
        let result = operation();
        self.timings.verify = started.elapsed();
        println!("Time taken to verify: {:?}", self.timings.verify);

        match (policy, result) {
            (_, Ok(())) => {
                println!("{VERIFY_STATUS_PASS}");
                Ok(VerifyOutcome::Verified)
            }
            (VerifyPolicy::Strict, Err(error)) => {
                println!("{VERIFY_STATUS_FAIL}");
                Err(error)
            }
            (VerifyPolicy::KnownIssue { message }, Err(error)) => {
                eprintln!("{KNOWN_ISSUE_WARNING_SENTINEL} {message}: {error}");
                println!("{VERIFY_STATUS_KNOWN_ISSUE}");
                Ok(VerifyOutcome::KnownIssue)
            }
        }
    }

    pub fn finish(self) -> RunReport {
        RunReport {
            timings: self.timings,
        }
    }
}

pub fn report_cuda_memory(free_bytes: usize, total_bytes: usize) {
    const BYTES_PER_MIB: f64 = 1024.0 * 1024.0;

    println!(
        "Cuda Memory used: {:.2}MB, Free: {:.2}MB",
        total_bytes.saturating_sub(free_bytes) as f64 / BYTES_PER_MIB,
        free_bytes as f64 / BYTES_PER_MIB
    );
}

pub fn report_memory_profile(profile: MemoryProfile, num_ldes: [usize; 4]) {
    println!("Memory profile: {profile}");
    println!(
        "num_ldes: [{}, {}, {}, {}]",
        num_ldes[0], num_ldes[1], num_ldes[2], num_ldes[3]
    );
}

pub fn report_circuit_size(degree_bits: usize, gates: usize) {
    println!("degree: {degree_bits}");
    println!("gates: {gates}");
}

pub fn report_proof_size(bytes: usize) {
    println!("Proof size: {bytes} bytes");
}

pub fn report_public_input_count(count: usize) {
    println!("Public inputs: {count}");
}
