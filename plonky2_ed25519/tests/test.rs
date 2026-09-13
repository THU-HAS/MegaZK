use std::fs::File;
use std::io::Read;

use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::fri::oracle::PolynomialBatch;
use plonky2::iop::witness;
use plonky2::plonk::config::PoseidonGoldilocksConfig;
use plonky2::util::timing::TimingTree;

#[test]
fn witness_check() {
    let mut file = File::open("witness.json").unwrap();
    let mut witness_str = String::new();
    file.read_to_string(&mut witness_str).unwrap();
    let witness: Vec<u64> = serde_json::from_str(&witness_str).expect("Failed to deserialize data");

    let mut file = File::open("witness_ref.json").unwrap();
    let mut witness_ref_str = String::new();
    file.read_to_string(&mut witness_ref_str).unwrap();
    let witness_ref: Vec<Vec<u64>> = serde_json::from_str(&witness_ref_str).expect("Failed to deserialize data");

    let witness_ref = witness_ref.concat();

    assert_eq!(witness.len(), witness_ref.len());
    for i in 0..witness.len() {
        assert_eq!(witness[i], witness_ref[i], "different at {:?}", i);
    }

}

#[test]
fn cm0_check() {
    let mut file = File::open("cm0data_ref.json").unwrap();
    let mut witness_str = String::new();
    file.read_to_string(&mut witness_str).unwrap();
    let witness: Vec<u64> = serde_json::from_str(&witness_str).expect("Failed to deserialize data");

    let mut file = File::open("cm0data_ref1.json").unwrap();
    let mut witness_ref_str = String::new();
    file.read_to_string(&mut witness_ref_str).unwrap();
    let witness_ref: Vec<u64> = serde_json::from_str(&witness_ref_str).expect("Failed to deserialize data");

    // let witness_ref = witness_ref.concat();

    assert_eq!(witness.len(), witness_ref.len());
    for i in 0..witness.len() {
        assert_eq!(witness[i], witness_ref[i], "different at {:?}", i);
    }

}

#[test]
fn zs_check() {
    let mut file = File::open("zs.json").unwrap();
    let mut witness_str = String::new();
    file.read_to_string(&mut witness_str).unwrap();
    let witness: Vec<u64> = serde_json::from_str(&witness_str).expect("Failed to deserialize data");

    let mut file = File::open("zs_ref.json").unwrap();
    let mut witness_ref_str = String::new();
    file.read_to_string(&mut witness_ref_str).unwrap();
    let witness_ref: Vec<u64> = serde_json::from_str(&witness_ref_str).expect("Failed to deserialize data");

    // let witness_ref = witness_ref.concat();

    assert_eq!(witness.len(), witness_ref.len());
    for i in 0..witness.len() {
        assert_eq!(witness[i], witness_ref[i], "different at {:?}", i);
    }

}

#[test]
fn qs_check() {
    let mut file = File::open("qs.json").unwrap();
    let mut witness_str = String::new();
    file.read_to_string(&mut witness_str).unwrap();
    // let witness: Vec<Vec<u64>> = serde_json::from_str(&witness_str).expect("Failed to deserialize data");
    let witness: Vec<u64> = serde_json::from_str(&witness_str).expect("Failed to deserialize data");
    // let witness = witness.concat();

    let mut file = File::open("qs_ref.json").unwrap();
    let mut witness_ref_str = String::new();
    file.read_to_string(&mut witness_ref_str).unwrap();
    let witness_ref: Vec<Vec<u64>> = serde_json::from_str(&witness_ref_str).expect("Failed to deserialize data");
    let witness_ref = witness_ref.concat();

    assert_eq!(witness.len(), witness_ref.len());
    for i in 0..witness.len() {
        assert_eq!(witness[i], witness_ref[i], "different at {:?}", i);
    }

}

