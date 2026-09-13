use anyhow::Result;
#[cfg(feature = "gpu")]
use cudart::device::set_device;
#[cfg(feature = "gpu")]
use cudart::memory::memory_get_info;
use plonky2::{
    field::{secp256k1_scalar::Secp256K1Scalar, types::Sample},
    iop::witness::PartialWitness,
    plonk::{
        circuit_builder::CircuitBuilder,
        circuit_data::CircuitConfig,
        config::{GenericConfig, PoseidonGoldilocksConfig},
    },
};
use plonky2_ecdsa::{
    curve::{
        curve_types::{Curve, CurveScalar},
        ecdsa::{sign_message_with_rng, ECDSAPublicKey, ECDSASecretKey, ECDSASignature},
        secp256k1::Secp256K1,
    },
    gadgets::{
        curve::CircuitBuilderCurve,
        ecdsa::{verify_message_circuit, ECDSAPublicKeyTarget, ECDSASignatureTarget},
        nonnative::CircuitBuilderNonNative,
    },
};
#[cfg(feature = "gpu")]
use plonky2_gpu_bench::report_cuda_memory;
use plonky2_gpu_bench::{
    report_circuit_size, report_memory_profile, report_proof_size, report_public_input_count,
    BenchSession, MemoryProfile, VerifyPolicy,
};
use rand::{rngs::StdRng, SeedableRng};
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
struct Cli {
    #[structopt(long)]
    baseline: bool,

    #[structopt(long, default_value = "15")]
    num: u32,

    #[structopt(long, default_value = "compressed")]
    memory_profile: MemoryProfile,

    #[structopt(
        long,
        default_value = "5eed5eed5eed5eed",
        parse(try_from_str = parse_hex_u64)
    )]
    seed: u64,
}

fn parse_hex_u64(value: &str) -> std::result::Result<u64, std::num::ParseIntError> {
    u64::from_str_radix(value.trim_start_matches("0x"), 16)
}

fn main() -> Result<()> {
    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;
    type F = <C as GenericConfig<D>>::F;
    type Curve = Secp256K1;

    let cli = Cli::from_args();
    let baseline = cli.baseline;
    #[cfg(not(feature = "gpu"))]
    if !baseline {
        anyhow::bail!("GPU proving is unavailable; rebuild with --features gpu");
    }
    let mut session = BenchSession::new();
    let num = cli.num;
    let num_ldes = cli.memory_profile.lde_counts([86, 136, 20, 16]);
    report_memory_profile(cli.memory_profile, num_ldes);
    println!("ECDSA seed: {:016x}", cli.seed);
    let mut rng = StdRng::seed_from_u64(cli.seed);
    // let num_ldes = [0, 68, 0, 0];
    // let num_ldes = [0, 136, 0, 0];
    // let num_ldes = [43, 136, 0, 0];
    // let num_ldes = [86, 136, 0, 0];
    // let num_ldes = [86, 136, 20, 16];

    let config = CircuitConfig::standard_ecc_config();
    let pw = PartialWitness::new();
    let mut builder = CircuitBuilder::<F, D>::new(config);

    for _i in 0..num {
        let msg = Secp256K1Scalar::sample(&mut rng);
        let msg_target = builder.constant_nonnative(msg);
        let sk = ECDSASecretKey::<Curve>(Secp256K1Scalar::sample(&mut rng));
        let pk = ECDSAPublicKey((CurveScalar(sk.0) * Curve::GENERATOR_PROJECTIVE).to_affine());
        let pk_target = ECDSAPublicKeyTarget(builder.constant_affine_point(pk.0));
        let sig = sign_message_with_rng(msg, sk, &mut rng);
        let ECDSASignature { r, s } = sig;
        let r_target = builder.constant_nonnative(r);
        let s_target = builder.constant_nonnative(s);
        let sig_target = ECDSASignatureTarget {
            r: r_target,
            s: s_target,
        };
        verify_message_circuit(&mut builder, msg_target, sig_target, pk_target);
    }

    let gates = builder.num_gates();

    if baseline {
        let data = session.build(|| builder.build::<C>());
        report_circuit_size(data.common.degree_bits(), gates);
        let proof = session.prove(|| data.prove(pw))?;
        report_proof_size(proof.to_bytes().len());
        report_public_input_count(proof.public_inputs.len());
        session.verify(VerifyPolicy::Strict, || data.verify(proof))?;
    } else {
        #[cfg(feature = "gpu")]
        {
            set_device(0)?;
            let input_targets = Vec::new();
            let inputs = Vec::new();
            let mut data = session.build(|| builder.build_gpu::<C>(&input_targets, num_ldes));
            let (free, total) = memory_get_info()?;
            report_cuda_memory(free, total);
            report_circuit_size(data.common.degree_bits(), gates);

            let proof = session.prove(|| data.prove(&inputs))?;
            report_proof_size(proof.to_bytes().len());
            report_public_input_count(proof.public_inputs.len());
            session.verify(VerifyPolicy::Strict, || data.verify(proof))?;
        }
        #[cfg(not(feature = "gpu"))]
        {
            anyhow::bail!("GPU proving is unavailable; rebuild with --features gpu");
        }
    }
    Ok(())
}
