use anyhow::Result;
use log::{Level, LevelFilter};
#[cfg(feature = "gpu")]
use plonky2::field::types::Field;
#[cfg(feature = "gpu")]
use plonky2::iop::target::Target;
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::CircuitConfig;
use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
use plonky2::util::timing::TimingTree;
#[cfg(feature = "gpu")]
use plonky2_gpu_bench::report_cuda_memory;
use plonky2_gpu_bench::{
    report_circuit_size, report_memory_profile, report_proof_size, report_public_input_count,
    BenchSession, MemoryProfile, VerifyPolicy,
};
use plonky2_sha256::circuit::{array_to_bits, make_circuits};
use sha2::{Digest, Sha256};
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

    #[structopt(long, default_value = "120")]
    num: u32,

    #[structopt(long, default_value = "compressed")]
    memory_profile: MemoryProfile,
}

pub fn prove_sha256(msg: &[u8]) -> Result<()> {
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

    if baseline {
        let mut hasher = Sha256::new();
        hasher.update(msg);
        let hash = hasher.finalize();
        // println!("Hash: {:#04X}", hash);

        let msg_bits = array_to_bits(msg);
        let len = msg.len() * 8;
        println!("block count: {}", (len + 65 + 511) / 512);
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        let mut builder = CircuitBuilder::<F, D>::new(CircuitConfig::standard_recursion_config());
        let mut pw = PartialWitness::new();
        for _i in 0..num {
            let targets = make_circuits(&mut builder, len as u64);
            for i in 0..len {
                pw.set_bool_target(targets.message[i], msg_bits[i]);
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
        println!("Constructing inner proof with {gates} gates");
        let data = session.build(|| builder.build::<C>());
        report_circuit_size(data.common.degree_bits(), gates);
        let timing = TimingTree::new("prove", Level::Debug);
        let proof = session.prove(|| data.prove(pw))?;
        report_proof_size(proof.to_bytes().len());
        report_public_input_count(proof.public_inputs.len());
        timing.print();

        let timing = TimingTree::new("verify", Level::Debug);
        session.verify(VerifyPolicy::Strict, || data.verify(proof))?;
        timing.print();

        Ok(())
    } else {
        #[cfg(feature = "gpu")]
        {
            let mut hasher = Sha256::new();
            hasher.update(msg);
            let hash = hasher.finalize();
            // println!("Hash: {:#04X}", hash);

            let msg_bits = array_to_bits(msg);
            let len = msg.len() * 8;
            println!("block count: {}", (len + 65 + 511) / 512);
            const D: usize = 2;
            type C = PoseidonGoldilocksConfig;
            type F = <C as GenericConfig<D>>::F;
            let mut builder =
                CircuitBuilder::<F, D>::new(CircuitConfig::standard_recursion_config());
            let mut input_targets = Vec::<Target>::new();
            let mut inputs = Vec::<F>::new();

            for _i in 0..num {
                let targets = make_circuits(&mut builder, len as u64);
                for i in 0..len {
                    input_targets.push(targets.message[i].target);
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
            println!("Constructing inner proof with {gates} gates");
            set_device(0)?;
            let mut data = session.build(|| builder.build_gpu::<C>(&input_targets, num_ldes));
            let (free, total) = memory_get_info()?;
            report_cuda_memory(free, total);
            report_circuit_size(data.common.degree_bits(), gates);

            let timing = TimingTree::new("prove", Level::Debug);
            let proof = session.prove(|| data.prove(&inputs))?;
            report_proof_size(proof.to_bytes().len());
            report_public_input_count(proof.public_inputs.len());
            timing.print();

            let timing = TimingTree::new("verify", Level::Debug);
            session.verify(VerifyPolicy::Strict, || data.verify(proof))?;
            timing.print();

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
    builder.filter_level(LevelFilter::Debug);
    builder.try_init()?;

    const MSG_SIZE: usize = 128;
    // const MSG_SIZE: usize = 8000;
    let mut msg = vec![0; MSG_SIZE as usize];
    for i in 0..MSG_SIZE - 1 {
        msg[i] = i as u8;
    }
    prove_sha256(&msg)
}
