#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
use anyhow::Result;
use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;
#[cfg(feature = "gpu")]
use plonky2::iop::target::Target;
use plonky2::iop::witness::PartialWitness;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::CircuitConfig;
use plonky2::plonk::config::{GenericConfig, Hasher, PoseidonGoldilocksConfig};
use plonky2_ed25519::curve::eddsa::{SAMPLE_MSG1, SAMPLE_PK1, SAMPLE_SIG1};
#[cfg(feature = "gpu")]
use plonky2_ed25519::gadgets::eddsa::set_inputs;
use plonky2_ed25519::gadgets::eddsa::{fill_circuits, make_verify_circuits};
#[cfg(feature = "gpu")]
use plonky2_gpu_bench::report_cuda_memory;
use plonky2_gpu_bench::{
    report_circuit_size, report_memory_profile, report_proof_size, report_public_input_count,
    BenchSession, MemoryProfile, VerifyOutcome, VerifyPolicy,
};

#[cfg(feature = "gpu")]
use cudart::device::set_device;
#[cfg(feature = "gpu")]
use cudart::memory::memory_get_info;

use structopt::StructOpt;

#[derive(Debug, StructOpt)]
struct Cli {
    #[structopt(long)]
    baseline: bool,

    #[structopt(long, default_value = "10")]
    num: u32,

    #[structopt(long, default_value = "compressed")]
    memory_profile: MemoryProfile,
}

const VERIFY_POLICY: VerifyPolicy = VerifyPolicy::KnownIssue {
    message: "Ed25519 proof generation succeeded, but verification is experimental",
};

fn prove_ed25519<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>(
    msg: &[u8],
    sigv: &[u8],
    pkv: &[u8],
    // ) -> Result<ProofTuple<F, C, D>>
) -> Result<()>
where
    [(); C::Hasher::HASH_SIZE]:,
{
    let cli = Cli::from_args();
    let baseline = cli.baseline;
    #[cfg(not(feature = "gpu"))]
    if !baseline {
        anyhow::bail!("GPU proving is unavailable; rebuild with --features gpu");
    }
    let mut session = BenchSession::new();
    let num = cli.num;
    let num_ldes = cli.memory_profile.lde_counts([87, 234, 20, 16]);
    report_memory_profile(cli.memory_profile, num_ldes);
    // let num_ldes = [0, 117, 0, 0];
    // let num_ldes = [0, 234, 0, 0];
    // let num_ldes = [43, 234, 0, 0];
    // let num_ldes = [87, 234, 0, 0];
    // let num_ldes = [87, 234, 20, 16];

    if baseline {
        let mut builder = CircuitBuilder::<F, D>::new(CircuitConfig::wide_ecc_config());
        let mut pw = PartialWitness::new();
        for _i in 0..num {
            let targets = make_verify_circuits(&mut builder, msg.len());
            fill_circuits::<F, D>(&mut pw, msg, sigv, pkv, &targets);
        }
        let gates = builder.num_gates();
        println!("Building ed25519 circuit with {gates:?} gates");
        let data_old = session.build(|| builder.build::<C>());
        report_circuit_size(data_old.common.degree_bits(), gates);
        // println!("Cap: {:?}", data_old.verifier_only.constants_sigmas_cap);

        let proof = session.prove(|| data_old.prove(pw))?;
        report_proof_size(proof.to_bytes().len());
        report_public_input_count(proof.public_inputs.len());
        let outcome = session.verify(VERIFY_POLICY, || data_old.verify(proof))?;
        if outcome == VerifyOutcome::Verified {
            println!("Ed25519 proof verified successfully.");
        }
    } else {
        #[cfg(feature = "gpu")]
        {
            set_device(0)?;
            let mut builder = CircuitBuilder::<F, D>::new(CircuitConfig::wide_ecc_config());
            let mut input_targets: Vec<Target> = Vec::<Target>::new();
            let mut inputs: Vec<F> = Vec::new();

            for _i in 0..num {
                let targets = make_verify_circuits(&mut builder, msg.len());
                set_inputs::<F, D>(&mut input_targets, &mut inputs, msg, sigv, pkv, &targets);
            }
            let gates = builder.num_gates();
            println!("Building ed25519 circuit with {gates:?} gates");
            let mut data = session.build(|| builder.build_gpu::<C>(&input_targets, num_ldes));
            report_circuit_size(data.common.degree_bits(), gates);
            // println!("Cap: {:?}", data.verifier_only.constants_sigmas_cap);

            let (free, total) = memory_get_info()?;
            report_cuda_memory(free, total);

            let proof = session.prove(|| data.prove(&inputs))?;
            report_proof_size(proof.to_bytes().len());
            report_public_input_count(proof.public_inputs.len());
            let outcome = session.verify(VERIFY_POLICY, || data.verify(proof))?;
            if outcome == VerifyOutcome::Verified {
                println!("Ed25519 proof verified successfully.");
            }
        }
        #[cfg(not(feature = "gpu"))]
        {
            anyhow::bail!("GPU proving is unavailable; rebuild with --features gpu");
        }
    }

    // Ok((proof, data.verifier_only, data.common))
    Ok(())
}

pub fn benchmark() -> Result<()> {
    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;
    type F = <C as GenericConfig<D>>::F;
    prove_ed25519::<F, C, D>(
        SAMPLE_MSG1.as_bytes(),
        SAMPLE_SIG1.as_slice(),
        SAMPLE_PK1.as_slice(),
    )?;
    Ok(())
}

fn main() -> Result<()> {
    // benchmark();
    benchmark()
}
