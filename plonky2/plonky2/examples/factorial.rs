use core::marker::PhantomData;
use std::fs::File;
use std::io::{Read, Write};
use std::thread::sleep;

use anyhow::Result;
#[cfg(feature = "gpu")]
use cudart::device::set_device;
#[cfg(feature = "gpu")]
use cudart::memory::memory_get_info;
use plonky2::field::types::Field;
#[cfg(feature = "gpu")]
use plonky2::iop::target::Target;
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::{CircuitConfig, CircuitData};
use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
use plonky2::util::serialization::{DefaultGateSerializer, DefaultGeneratorSerializer};
#[cfg(feature = "gpu")]
use plonky2_gpu_bench::report_cuda_memory;
use plonky2_gpu_bench::{
    report_circuit_size, report_memory_profile, report_proof_size, report_public_input_count,
    BenchSession, MemoryProfile, VerifyPolicy,
};
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
struct Cli {
    // 默认false，加--baseline为true
    #[structopt(long)]
    baseline: bool,

    #[structopt(long, default_value = "524289")]
    num: u32,

    #[structopt(long, default_value = "compressed")]
    memory_profile: MemoryProfile,
}

/// An example of using Plonky2 to prove a statement of the form
/// "I know n * (n + 1) * ... * (n + 99)".
/// When n == 1, this is proving knowledge of 100!.
fn main() -> Result<()> {
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
    let mut pw = PartialWitness::new();
    #[cfg(feature = "gpu")]
    let mut input_targets = Vec::<Target>::new();
    #[cfg(feature = "gpu")]
    let mut inputs = Vec::<F>::new();

    for _ in 0..64 {
        // The arithmetic circuit.
        let initial = builder.add_virtual_target();
        let mut cur_target = initial;

        for i in 2..num {
            let i_target = builder.constant(F::from_canonical_u32(i));
            cur_target = builder.mul(cur_target, i_target);
        }

        // Public inputs are the initial value (provided below) and the result (which is generated).
        builder.register_public_input(initial);
        builder.register_public_input(cur_target);

        pw.set_target(initial, F::ONE);

        #[cfg(feature = "gpu")]
        {
            input_targets.push(initial);
            inputs.push(F::ONE);
        }
    }

    let gates = builder.num_gates();
    println!("Constructing inner proof with {gates} gates");

    if baseline == true {
        let data = session.build(|| builder.build::<C>());
        report_circuit_size(data.common.degree_bits(), gates);

        // let gate_serializer = DefaultGateSerializer;
        // let generator_serializer = DefaultGeneratorSerializer::<C, D> {
        //     _phantom: PhantomData,
        // };

        // let data_bytes = data
        //     .to_bytes(&gate_serializer, &generator_serializer)
        //     .map_err(|_| anyhow::Error::msg("CircuitData serialization failed."))?;

        // let mut file = File::create("circuit_data.bin")?;
        // file.write_all(&data_bytes)?;
        // println!("Circuit data saved to circuit_data.bin");

        // let mut file = File::open("circuit_data.bin")?;
        // let mut data_bytes = Vec::new();
        // file.read_to_end(&mut data_bytes)?;

        // let data = CircuitData::<F, C, D>::from_bytes(
        //     &data_bytes,
        //     &gate_serializer,
        //     &generator_serializer,
        // )
        // .map_err(|_| anyhow::Error::msg("CircuitData deserialization failed."))?;

        let proof = session.prove(|| data.prove(pw))?;
        report_proof_size(proof.to_bytes().len());
        report_public_input_count(proof.public_inputs.len());

        println!(
            "Factorial starting at {} is {}",
            proof.public_inputs[0], proof.public_inputs[1]
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

            let proof = session.prove(|| data.prove(&inputs))?;
            report_proof_size(proof.to_bytes().len());
            report_public_input_count(proof.public_inputs.len());
            println!(
                "Factorial starting at {} is {}",
                proof.public_inputs[0], proof.public_inputs[1]
            );
            session.verify(VerifyPolicy::Strict, || data.verify(proof))?;
            Ok(())
        }
        #[cfg(not(feature = "gpu"))]
        {
            anyhow::bail!("GPU proving is unavailable; rebuild with --features gpu");
        }
    }
}
