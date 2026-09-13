use boojum::field::goldilocks::GoldilocksField as GoldilocksFieldBoojum;
use boojum::field::Field;

#[derive(Clone)]
pub struct Powers {
    base: GoldilocksFieldBoojum,
    current: GoldilocksFieldBoojum,
}

impl Iterator for Powers {
    type Item = GoldilocksFieldBoojum;

    fn next(&mut self) -> Option<GoldilocksFieldBoojum> {
        let result = self.current;
        self.current *= self.base;
        Some(result)
    }
}

pub fn powers(x: GoldilocksFieldBoojum) -> Powers {
    Powers {
        base: x,
        current: GoldilocksFieldBoojum::ONE,
    }
}
