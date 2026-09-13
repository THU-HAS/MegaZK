//! One setup-time device scratch block, viewed as witgen SoA or as cm1–cm3 + FRI.
//!
//! Peak device memory is `max(captured witgen SoA, delayed commit + FRI)` when the
//! graph is destroyed before FRI, or the sum when a live graph keeps SoA and FRI
//! is allocated extra. Views do not `cudaFree`; `GpuWorkspace` owns the blocks.

use std::mem::{self, ManuallyDrop};
use std::ops::{Deref, DerefMut};

use boojum::field::goldilocks::GoldilocksField as GoldilocksFieldBoojum;
use cudart::device::device_synchronize;
use cudart::memory::DeviceAllocation;
use cudart::slice::{CudaSlice, CudaSliceMut};

const SCRATCH_ALIGN: usize = 256;

/// Headroom when deciding whether extra FRI can sit beside a live graph.
pub(crate) const FRI_KEEP_SOA_SLACK: usize = 256 * 1024 * 1024;

pub(crate) fn align_up(bytes: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two());
    (bytes + align - 1) & !(align - 1)
}

/// Non-owning device buffer. `Drop` does not call `cudaFree`.
pub(crate) struct DeviceView<T> {
    inner: ManuallyDrop<DeviceAllocation<T>>,
}

impl<T> DeviceView<T> {
    pub(crate) fn empty() -> Self {
        unsafe { Self::from_raw(std::ptr::null_mut(), 0) }
    }

    /// # Safety
    /// `ptr` must be a valid device pointer for `len` elements, or null when `len == 0`.
    /// The allocation must outlive this view and must not be freed through this view.
    pub(crate) unsafe fn from_raw(ptr: *mut T, len: usize) -> Self {
        Self {
            inner: ManuallyDrop::new(DeviceAllocation::from_raw_parts(ptr, len)),
        }
    }
}

impl<T> Deref for DeviceView<T> {
    type Target = DeviceAllocation<T>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T> DerefMut for DeviceView<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<T> CudaSlice<T> for DeviceView<T> {
    unsafe fn as_slice(&self) -> &[T] {
        CudaSlice::as_slice(&*self.inner)
    }
}

impl<T> CudaSliceMut<T> for DeviceView<T> {
    unsafe fn as_mut_slice(&mut self) -> &mut [T] {
        CudaSliceMut::as_mut_slice(&mut *self.inner)
    }
}

pub(crate) struct ScratchCursor {
    base: *mut u8,
    len: usize,
    offset: usize,
}

impl ScratchCursor {
    pub(crate) fn new(scratch: &mut DeviceAllocation<u8>) -> Self {
        Self {
            base: scratch.as_mut_ptr(),
            len: scratch.len(),
            offset: 0,
        }
    }

    pub(crate) fn take<T>(&mut self, count: usize) -> DeviceView<T> {
        let size = count
            .checked_mul(mem::size_of::<T>())
            .expect("overlay region size overflow");
        let start = align_up(self.offset, SCRATCH_ALIGN);
        let end = start.checked_add(size).expect("overlay region size overflow");
        assert!(
            end <= self.len,
            "overlay scratch overflow: need {end} bytes, have {}",
            self.len
        );
        let ptr = unsafe { self.base.add(start) as *mut T };
        self.offset = end;
        unsafe { DeviceView::from_raw(ptr, count) }
    }
}

pub(crate) struct ScratchSizer {
    offset: usize,
}

impl ScratchSizer {
    pub(crate) fn new() -> Self {
        Self { offset: 0 }
    }

    pub(crate) fn add<T>(&mut self, count: usize) {
        let size = count
            .checked_mul(mem::size_of::<T>())
            .expect("overlay region size overflow");
        self.offset = align_up(self.offset, SCRATCH_ALIGN);
        self.offset = self
            .offset
            .checked_add(size)
            .expect("overlay region size overflow");
    }

    pub(crate) fn bytes(self) -> usize {
        align_up(self.offset, SCRATCH_ALIGN)
    }
}

pub(crate) fn merkle_tree_len(degree: usize, rate_bits: usize) -> usize {
    8 * degree << rate_bits
}

pub(crate) struct FriCommitPartLens {
    pub digests_len: usize,
    pub leaves_len: usize,
    pub folded_len: usize,
}

pub(crate) fn fri_commit_part_lens(
    degree: usize,
    extension_degree: usize,
    rate_bits: usize,
    reduction_arity_bits: &[usize],
) -> Vec<FriCommitPartLens> {
    let mut digests_len = 8 * degree << rate_bits;
    let mut leaves_len = degree * extension_degree << rate_bits;
    let mut parts = Vec::with_capacity(reduction_arity_bits.len());
    for &arity_bits in reduction_arity_bits {
        digests_len >>= arity_bits;
        let folded_len = leaves_len >> arity_bits;
        parts.push(FriCommitPartLens {
            digests_len,
            leaves_len,
            folded_len,
        });
        leaves_len = folded_len;
    }
    parts
}

pub(crate) fn add_delayed_commit_and_fri(
    sizer: &mut ScratchSizer,
    degree: usize,
    extension_degree: usize,
    rate_bits: usize,
    reduction_arity_bits: &[usize],
) {
    let tree_len = merkle_tree_len(degree, rate_bits);
    for _ in 0..3 {
        sizer.add::<GoldilocksFieldBoojum>(tree_len);
    }
    for part in fri_commit_part_lens(degree, extension_degree, rate_bits, reduction_arity_bits) {
        sizer.add::<GoldilocksFieldBoojum>(part.digests_len);
        sizer.add::<GoldilocksFieldBoojum>(part.leaves_len);
        sizer.add::<GoldilocksFieldBoojum>(part.leaves_len);
        sizer.add::<GoldilocksFieldBoojum>(part.folded_len);
    }
}

pub(crate) fn post_witgen_bytes(
    degree: usize,
    extension_degree: usize,
    rate_bits: usize,
    reduction_arity_bits: &[usize],
) -> usize {
    let mut sizer = ScratchSizer::new();
    add_delayed_commit_and_fri(
        &mut sizer,
        degree,
        extension_degree,
        rate_bits,
        reduction_arity_bits,
    );
    sizer.bytes()
}

#[allow(dead_code)]
pub(crate) fn witgen_bytes(
    read_len: usize,
    witness_len: usize,
    rep_map_len: usize,
    params_len: usize,
) -> usize {
    let mut sizer = ScratchSizer::new();
    sizer.add::<u32>(read_len);
    sizer.add::<GoldilocksFieldBoojum>(witness_len);
    sizer.add::<u32>(rep_map_len);
    sizer.add::<usize>(params_len);
    sizer.bytes()
}

/// Graph-captured SoA only: generator kernels read `read_map` / `witness` / `params`.
/// The scatter `representative_map` is a separate owned allocation so it can be
/// freed after the last replay without moving captured addresses.
pub(crate) fn witgen_captured_bytes(read_len: usize, witness_len: usize, params_len: usize) -> usize {
    let mut sizer = ScratchSizer::new();
    sizer.add::<u32>(read_len);
    sizer.add::<GoldilocksFieldBoojum>(witness_len);
    sizer.add::<usize>(params_len);
    sizer.bytes()
}

pub(crate) fn ensure_scratch_bytes(scratch: &mut DeviceAllocation<u8>, needed: usize) {
    if scratch.len() >= needed {
        return;
    }
    println!(
        "Grew overlay scratch from {} MiB to {} MiB (captured SoA; FRI is extra after replay)",
        scratch.len() as f64 / (1024.0 * 1024.0),
        needed as f64 / (1024.0 * 1024.0)
    );
    // Witgen views are unbound by the caller. Delayed commit/FRI views are still
    // unallocated at this point. Synchronize before cudaFree of the old block so a
    // later first-alloc cannot alias in-flight constants NTT/LDE.
    device_synchronize().unwrap();
    let new = DeviceAllocation::<u8>::alloc(needed).unwrap();
    *scratch = new;
    device_synchronize().unwrap();
}
