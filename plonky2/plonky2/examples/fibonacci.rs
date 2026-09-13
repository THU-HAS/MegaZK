use std::thread::sleep;

use anyhow::Result;
#[cfg(feature = "gpu")]
use cudart::device::set_device;
#[cfg(feature = "gpu")]
use cudart::memory::memory_get_info;
use log::{Level, LevelFilter};
use plonky2::field::types::Field;
#[cfg(feature = "gpu")]
use plonky2::iop::target::Target;
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::CircuitConfig;
use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
#[cfg(feature = "gpu")]
use plonky2_gpu_bench::report_cuda_memory;
use plonky2_gpu_bench::{
    report_circuit_size, report_memory_profile, report_proof_size, report_public_input_count,
    BenchSession, MemoryProfile, VerifyPolicy,
};
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
struct Cli {
    #[structopt(long)]
    baseline: bool,

    #[structopt(long, default_value = "524289")]
    num: u32,

    #[structopt(long, default_value = "compressed")]
    memory_profile: MemoryProfile,
}

/// An example of using Plonky2 to prove a statement of the form
/// "I know the 100th element of the Fibonacci sequence, starting with constants a and b."
/// When a == 0 and b == 1, this is proving knowledge of the 100th (standard) Fibonacci number.
fn main() -> Result<()> {
    let mut builder = env_logger::Builder::from_default_env();
    builder.format_timestamp(None);
    builder.filter_level(LevelFilter::Debug);
    builder.try_init()?;

    let cli = Cli::from_args();
    let baseline = cli.baseline;
    #[cfg(not(feature = "gpu"))]
    if !baseline {
        anyhow::bail!("GPU proving is unavailable; rebuild with --features gpu");
    }
    let mut session = BenchSession::new();
    let num: u32 = cli.num;
    let num_ldes = cli.memory_profile.lde_counts([84, 135, 20, 16]);
    report_memory_profile(cli.memory_profile, num_ldes);
    // let num_ldes = [0, 65, 0, 0];
    // let num_ldes = [0, 135, 0, 0];
    // let num_ldes = [42, 135, 0, 0];
    // let num_ldes = [84, 135, 0, 0];
    // let num_ldes = [84, 135, 20, 16];

    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;
    type F = <C as GenericConfig<D>>::F;

    let config = CircuitConfig::standard_recursion_config();
    let mut builder = CircuitBuilder::<F, D>::new(config);

    // Provide initial values.
    let mut pw = PartialWitness::new();
    #[cfg(feature = "gpu")]
    let mut input_targets = Vec::<Target>::new();
    #[cfg(feature = "gpu")]
    let mut inputs = Vec::<F>::new();

    for _ in 0..64 {
        // The arithmetic circuit.
        let initial_a = builder.add_virtual_target();
        let initial_b = builder.add_virtual_target();
        let mut prev_target = initial_a;
        let mut cur_target = initial_b;
        for _ in 0..num {
            let temp = builder.add(prev_target, cur_target);
            prev_target = cur_target;
            cur_target = temp;
        }

        // Public inputs are the two initial values (provided below) and the result (which is generated).
        builder.register_public_input(initial_a);
        builder.register_public_input(initial_b);
        builder.register_public_input(cur_target);

        pw.set_target(initial_a, F::ZERO);
        pw.set_target(initial_b, F::ONE);

        #[cfg(feature = "gpu")]
        {
            input_targets.push(initial_a);
            input_targets.push(initial_b);
            inputs.push(F::ZERO);
            inputs.push(F::ONE);
        }
    }

    let gates = builder.num_gates();
    println!("Constructing inner proof with {gates} gates");

    if baseline == true {
        let data = session.build(|| builder.build::<C>());
        report_circuit_size(data.common.degree_bits(), gates);

        let proof = session.prove(|| data.prove(pw))?;
        report_proof_size(proof.to_bytes().len());
        report_public_input_count(proof.public_inputs.len());

        println!("Fibonacci term index: {}", u64::from(num) + 1);
        println!(
            "Fibonacci term mod |F| (starting with {}, {}) is: {}",
            proof.public_inputs[0], proof.public_inputs[1], proof.public_inputs[2]
        );

        session.verify(VerifyPolicy::Strict, || data.verify(proof))?;
        Ok(())
    } else {
        #[cfg(feature = "gpu")]
        {
            set_device(0)?;
            let mut data = session.build(|| builder.build_gpu::<C>(&input_targets, num_ldes));
            let (free, total) = memory_get_info()?;
            report_cuda_memory(free, total);
            report_circuit_size(data.common.degree_bits(), gates);

            println!("--Start to prove--");
            let proof = session.prove(|| data.prove(&inputs))?;
            report_proof_size(proof.to_bytes().len());
            report_public_input_count(proof.public_inputs.len());
            println!("Fibonacci term index: {}", u64::from(num) + 1);
            session.verify(VerifyPolicy::Strict, || data.verify(proof))?;
            Ok(())
        }
        #[cfg(not(feature = "gpu"))]
        {
            anyhow::bail!("GPU proving is unavailable; rebuild with --features gpu");
        }
    }
}
