use std::mem::{self, MaybeUninit};
use std::os::raw::c_void;
use std::ptr::{null, null_mut};

use cudart_sys::*;

use crate::execution::Dim3;
use crate::result::{CudaResult, CudaResultWrap};
use crate::stream::CudaStream;


/// # Example
/// ```ignore
/// let params = CudaKernelNodeParams::new(my_kernel as *const c_void)
///     .grid_dim(block_x)
///     .block_dim(128u32)
///     .arg(&witness)
///     .arg(&length);
/// let node = graph.add_kernel_node(&params.build(), &deps)?;
/// ```
pub struct CudaKernelNodeParams {
    func: *mut c_void,
    grid_dim: Dim3,
    block_dim: Dim3,
    shared_mem_bytes: u32,
    args: Vec<*mut c_void>,
}

impl CudaKernelNodeParams {
    pub fn new(func: *const c_void) -> Self {
        Self {
            func: func as *mut c_void,
            grid_dim: Dim3::default(),
            block_dim: Dim3::default(),
            shared_mem_bytes: 0,
            args: Vec::new(),
        }
    }

    pub fn grid_dim(mut self, grid_dim: impl Into<Dim3>) -> Self {
        self.grid_dim = grid_dim.into();
        self
    }

    pub fn block_dim(mut self, block_dim: impl Into<Dim3>) -> Self {
        self.block_dim = block_dim.into();
        self
    }

    pub fn shared_mem_bytes(mut self, bytes: u32) -> Self {
        self.shared_mem_bytes = bytes;
        self
    }

    pub fn arg<T>(mut self, value: &T) -> Self {
        self.args.push(value as *const T as *mut c_void);
        self
    }

    pub fn args(mut self, ptrs: &[*mut c_void]) -> Self {
        self.args.extend_from_slice(ptrs);
        self
    }

    pub fn build(&mut self) -> cudaKernelNodeParams {
        cudaKernelNodeParams {
            func: self.func,
            gridDim: self.grid_dim.into(),
            blockDim: self.block_dim.into(),
            sharedMemBytes: self.shared_mem_bytes,
            kernelParams: self.args.as_mut_ptr(),
            extra: null_mut(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CudaGraphNode {
    handle: cudaGraphNode_t,
}

impl CudaGraphNode {
    fn as_raw(&self) -> cudaGraphNode_t {
        self.handle
    }
}

pub struct CudaGraph {
    handle: cudaGraph_t,
}

impl CudaGraph {
    pub fn new() -> CudaResult<Self> {
        let mut graph = MaybeUninit::<cudaGraph_t>::uninit();
        unsafe {
            cudaGraphCreate(graph.as_mut_ptr(), 0)
                .wrap_maybe_uninit(graph)
                .map(|g| CudaGraph { handle: g })
        }
    }

    pub fn add_kernel_node(
        &mut self,
        params: &cudaKernelNodeParams,
        dependencies: &[CudaGraphNode],
    ) -> CudaResult<CudaGraphNode> {
        let mut node = MaybeUninit::<cudaGraphNode_t>::uninit();
        let dep_ptrs: Vec<cudaGraphNode_t> = dependencies.iter().map(|n| n.as_raw()).collect();
        let (deps_ptr, deps_len) = if dep_ptrs.is_empty() {
            (null(), 0)
        } else {
            (dep_ptrs.as_ptr(), dep_ptrs.len())
        };
        unsafe {
            cudaGraphAddKernelNode(
                node.as_mut_ptr(),
                self.handle,
                deps_ptr,
                deps_len,
                params as *const cudaKernelNodeParams,
            )
            .wrap_maybe_uninit(node)
            .map(|n| CudaGraphNode { handle: n })
        }
    }

    pub fn instantiate(&self) -> CudaResult<CudaGraphExec> {
        let mut exec = MaybeUninit::<cudaGraphExec_t>::uninit();
        unsafe {
            cudaGraphInstantiate(exec.as_mut_ptr(), self.handle, 0)
                .wrap_maybe_uninit(exec)
                .map(|e| CudaGraphExec { handle: e })
        }
    }

    pub fn destroy(self) -> CudaResult<()> {
        let handle = self.handle;
        mem::forget(self);
        unsafe { cudaGraphDestroy(handle).wrap() }
    }
}

impl Drop for CudaGraph {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { cudaGraphDestroy(self.handle).eprint_error_and_backtrace() };
        }
    }
}

pub struct CudaGraphExec {
    handle: cudaGraphExec_t,
}

impl CudaGraphExec {
    pub fn launch(&self, stream: &CudaStream) -> CudaResult<()> {
        unsafe { cudaGraphLaunch(self.handle, stream.handle).wrap() }
    }

    pub fn destroy(self) -> CudaResult<()> {
        let handle = self.handle;
        mem::forget(self);
        unsafe { cudaGraphExecDestroy(handle).wrap() }
    }
}

impl Drop for CudaGraphExec {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { cudaGraphExecDestroy(self.handle).eprint_error_and_backtrace() };
        }
    }
}