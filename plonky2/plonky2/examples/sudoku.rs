use anyhow::Result;
use plonky2::field::extension::Extendable;
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::Field;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::target::Target;
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::CircuitConfig;
use plonky2::plonk::config::PoseidonGoldilocksConfig;

fn change_type(v: usize) -> u32 {
    v as u32
}

fn NotEqual<F: RichField + Extendable<D>, const D: usize>(
    b: &mut CircuitBuilder<F, D>,
    x: Target,
    y: Target,
) {
    let k1 = b.sub(x, y);
    let z = b.inverse(k1);
    let k = b.mul(z, k1);
    b.assert_one(k);
}

fn AllDifferent<F: RichField + Extendable<D>, const D: usize>(
    b: &mut CircuitBuilder<F, D>,
    x: Vec<Target>,
) {
    for i in 0..x.len() {
        for j in (i + 1)..x.len() {
            NotEqual(b, x[i], x[j]);
        }
    }
}

fn main() -> Result<()> {
    // We have a public input unsolved_grid and a private input solved_grid.
    // We want to prove that unsolved_grid is a valid sudoku grid and that it is
    // a solution to solved_grid.
    // this sudoku version for 4*4 grid and it just looks if every row has 1, 2, 3, 4 permutation

    // standard proof setup
    let config = CircuitConfig::standard_recursion_config();
    type F = GoldilocksField;
    type C = PoseidonGoldilocksConfig;
    const D: usize = 2;
    let mut builder = CircuitBuilder::<F, D>::new(config);

    // The unsolved grid.
    let mut unsolved_grid: Vec<plonky2::iop::target::Target> = Vec::new();
    for _ in 0..4 {
        for _ in 0..4 {
            unsolved_grid.push(builder.add_virtual_target());
        }
    }
    for i in 0..4 {
        for j in 0..4 {
            builder.register_public_input(unsolved_grid[i * 4 + j]);
        }
    }

    // The solved grid.
    let mut solved_grid: Vec<plonky2::iop::target::Target> = Vec::new();
    for _ in 0..4 {
        for _ in 0..4 {
            solved_grid.push(builder.add_virtual_target());
        }
    }

    // check the solved grid is a valid solution
    // check all cells are in range
    for i in 0..4 {
        for j in 0..4 {
            let mut temp1 = builder.one();
            let mut temp2 = builder.sub(solved_grid[i * 4 + j], temp1);
            builder.range_check(temp2, 2);
            temp1 = builder.constant(F::from_canonical_u32(3));
            let mut temp3 = builder.sub(temp1, temp2);
            builder.range_check(temp3, 2);
        }
    }

    // check all rows are different
    for i in 0..4 {
        AllDifferent(&mut builder, solved_grid[(i * 4)..((i + 1) * 4)].to_vec());
    }

    // check both grids are compitable
    for i in 0..4 {
        for j in 0..4 {
            let temp5 = builder.sub(unsolved_grid[i * 4 + j], solved_grid[i * 4 + j]);
            let temp6 = builder.mul(unsolved_grid[i * 4 + j], temp5);
            builder.assert_zero(temp6);
        }
    }

    // generate circuit data
    let data = builder.build::<C>();
    let mut pw = PartialWitness::<F>::new();
    // Provide initial values.
    for i in 0..4 {
        for j in 0..4 {
            pw.set_target(
                solved_grid[i * 4 + j],
                F::from_canonical_u32((change_type(j).rem_euclid(4) + 1)),
            );
        }
    }
    for i in 0..4 {
        for j in 0..4 {
            if change_type(j + i).rem_euclid(4) == 3 {
                pw.set_target(unsolved_grid[i * 4 + j], F::ZERO);
            } else {
                pw.set_target(
                    unsolved_grid[i * 4 + j],
                    F::from_canonical_u32((change_type(j).rem_euclid(4) + 1)),
                );
            }
        }
    }

    // Generate proof
    let proof = data.prove(pw)?;
    data.verify(proof)?;
    Ok(())
}
