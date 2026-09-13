use anyhow::{Ok, Result};
#[cfg(feature = "gpu")]
use plonky2::field::types::Field;
#[cfg(feature = "gpu")]
use plonky2::iop::target::Target;
use plonky2::{
    field::extension::Extendable,
    hash::hash_types::RichField,
    iop::witness::{PartialWitness, WitnessWrite},
    plonk::{
        circuit_builder::CircuitBuilder,
        circuit_data::CircuitConfig,
        config::{GenericConfig, PoseidonGoldilocksConfig},
    },
};
#[cfg(feature = "gpu")]
use plonky2_gpu_bench::report_cuda_memory;
use plonky2_gpu_bench::{
    report_circuit_size, report_memory_profile, report_proof_size, report_public_input_count,
    BenchSession, MemoryProfile, VerifyPolicy,
};
use plonky2_sha512::circuit::{array_to_bits, make_circuits};
use sha2::{Digest, Sha512};
use std::thread::sleep;

#[cfg(feature = "gpu")]
use cudart::device::set_device;
#[cfg(feature = "gpu")]
use cudart::memory::memory_get_info;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
struct Cli {
    #[structopt(long)]
    baseline: bool,

    #[structopt(long, default_value = "60")]
    num: u32,

    #[structopt(long, default_value = "compressed")]
    memory_profile: MemoryProfile,
}

pub fn prove_sha512(msg: &[u8]) -> Result<()> {
    let cli = Cli::from_args();
    let baseline = cli.baseline;
    #[cfg(not(feature = "gpu"))]
    if !baseline {
        anyhow::bail!("GPU proving is unavailable; rebuild with --features gpu");
    }
    let mut session = BenchSession::new();
    let num = cli.num;
    let num_ldes = cli.memory_profile.lde_counts([84, 135, 20, 16]);
    report_memory_profile(cli.memory_profile, num_ldes);
    // let num_ldes = [0, 65, 0, 0];
    // let num_ldes = [0, 135, 0, 0];
    // let num_ldes = [42, 135, 0, 0];
    // let num_ldes = [84, 135, 0, 0];
    // let num_ldes = [84, 135, 20, 16];

    let mut hasher = Sha512::new();
    hasher.update(msg);
    let hash = hasher.finalize();

    let msg_bits = array_to_bits(msg);
    let len = msg.len() * 8;
    let block_count = (len + 129 + 1023) / 1024;
    println!("block count: {}", block_count);

    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;
    type F = <C as GenericConfig<D>>::F;
    let mut builder = CircuitBuilder::<F, D>::new(CircuitConfig::standard_recursion_config());
    let mut pw = PartialWitness::new();
    #[cfg(feature = "gpu")]
    let mut input_targets = Vec::<Target>::new();
    #[cfg(feature = "gpu")]
    let mut inputs = Vec::<F>::new();

    for _i in 0..num {
        let targets = make_circuits(&mut builder, len as u128);
        for i in 0..len {
            pw.set_bool_target(targets.message[i], msg_bits[i]);
            #[cfg(feature = "gpu")]
            input_targets.push(targets.message[i].target);
            #[cfg(feature = "gpu")]
            inputs.push(F::from_bool(msg_bits[i]));
        }
        let expected_res = array_to_bits(hash.as_slice());
        for i in 0..expected_res.len() {
            if expected_res[i] {
                builder.assert_one(targets.digest[i].target);
            } else {
                builder.assert_zero(targets.digest[i].target);
            }
        }
    }

    let gates = builder.num_gates();
    println!("Circuit has {gates} gates");

    if baseline {
        let data = session.build(|| builder.build::<C>());
        report_circuit_size(data.common.degree_bits(), gates);

        let proof = session.prove(|| data.prove(pw))?;
        report_proof_size(proof.to_bytes().len());
        report_public_input_count(proof.public_inputs.len());
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
            session.verify(VerifyPolicy::Strict, || data.verify(proof))?;
            Ok(())
        }
        #[cfg(not(feature = "gpu"))]
        {
            anyhow::bail!("GPU proving is unavailable; rebuild with --features gpu");
        }
    }
}

fn main() -> Result<()> {
    // Initialize logging
    let mut builder = env_logger::Builder::from_default_env();
    builder.format_timestamp(None);
    builder.try_init()?;

    const MSG_SIZE: usize = 128;
    let mut msg = vec![0; MSG_SIZE as usize];
    for i in 0..MSG_SIZE - 1 {
        msg[i] = i as u8;
    }
    prove_sha512(&msg)
}
