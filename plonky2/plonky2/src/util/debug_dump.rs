use std::convert::Infallible;
use std::fmt::Display;
#[cfg(feature = "gpu")]
use std::mem::size_of;
use std::path::{Component, Path, PathBuf};
use std::sync::OnceLock;
use std::{env, fs};

#[cfg(feature = "gpu")]
use cudart::memory::{memory_copy_async_w_offset, CudaHostAllocFlags, HostAllocation};
#[cfg(feature = "gpu")]
use cudart::slice::CudaSlice;
#[cfg(feature = "gpu")]
use cudart::stream::CudaStream;

const DUMP_DIR_ENV: &str = "PLONKY2_GPU_DUMP_DIR";
const DEFAULT_DUMP_DIR: &str = "plonky2-gpu-dumps";

struct DumpState {
    root: Option<PathBuf>,
    root_ready: OnceLock<bool>,
}

static DUMP_STATE: OnceLock<DumpState> = OnceLock::new();

pub(crate) fn enabled() -> bool {
    dump_root().is_some()
}

pub(crate) fn write_json<E: Display>(
    file_name: &str,
    capture_and_serialize: impl FnOnce() -> Result<Vec<u8>, E>,
) {
    write_bytes(file_name, capture_and_serialize);
}

pub(crate) fn write_text(file_name: &str, render: impl FnOnce() -> String) {
    write_bytes(file_name, || {
        Ok::<Vec<u8>, Infallible>(render().into_bytes())
    });
}

#[cfg(feature = "gpu")]
pub(crate) fn capture_cuda<T: Clone, S: CudaSlice<T> + ?Sized>(
    source: &S,
    stream: &CudaStream,
) -> anyhow::Result<Vec<T>> {
    capture_cuda_bytes(source, 0, source.len(), stream)
}

#[cfg(feature = "gpu")]
pub(crate) fn capture_cuda_bytes<T: Clone, S: CudaSlice<T> + ?Sized>(
    source: &S,
    source_offset_bytes: usize,
    length: usize,
    stream: &CudaStream,
) -> anyhow::Result<Vec<T>> {
    let copy_size = length
        .checked_mul(size_of::<T>())
        .ok_or_else(|| anyhow::anyhow!("debug dump copy size overflow"))?;
    let source_size = source
        .len()
        .checked_mul(size_of::<T>())
        .ok_or_else(|| anyhow::anyhow!("debug dump source size overflow"))?;
    let source_end = source_offset_bytes
        .checked_add(copy_size)
        .ok_or_else(|| anyhow::anyhow!("debug dump source range overflow"))?;
    anyhow::ensure!(
        source_end <= source_size,
        "debug dump source range {source_offset_bytes}..{source_end} exceeds {source_size} bytes"
    );

    let mut host = HostAllocation::<T>::alloc(length, CudaHostAllocFlags::DEFAULT)?;
    memory_copy_async_w_offset(&mut host, source, 0, source_offset_bytes, copy_size, stream)?;
    stream.synchronize()?;
    Ok(host.to_vec())
}

fn write_bytes<E: Display>(
    file_name: &str,
    capture_and_serialize: impl FnOnce() -> Result<Vec<u8>, E>,
) {
    let state = dump_state();
    let Some(root) = state.root.as_deref() else {
        return;
    };
    let Some(path) = dump_path(&root, file_name) else {
        return;
    };
    if !root_ready(state, root) {
        return;
    }

    let contents = match capture_and_serialize() {
        Ok(contents) => contents,
        Err(err) => {
            warn(format_args!("failed to prepare {}: {err}", path.display()));
            return;
        }
    };

    if let Err(err) = fs::write(&path, contents) {
        warn(format_args!("failed to write {}: {err}", path.display()));
    }
}

fn dump_state() -> &'static DumpState {
    DUMP_STATE.get_or_init(|| DumpState {
        root: resolve_dump_root(),
        root_ready: OnceLock::new(),
    })
}

fn dump_root() -> Option<&'static Path> {
    dump_state().root.as_deref()
}

fn resolve_dump_root() -> Option<PathBuf> {
    if let Some(configured) = env::var_os(DUMP_DIR_ENV) {
        if configured.is_empty() {
            warn(format_args!(
                "{DUMP_DIR_ENV} is empty; debug dumps are disabled"
            ));
            return None;
        }
        return resolve_root(PathBuf::from(configured));
    }

    if !cfg!(feature = "verbose") {
        return None;
    }

    let target_dir = env::var_os("CARGO_TARGET_DIR")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target"));
    resolve_root(target_dir.join(DEFAULT_DUMP_DIR))
}

fn root_ready(state: &DumpState, root: &Path) -> bool {
    *state
        .root_ready
        .get_or_init(|| match fs::create_dir_all(root) {
            Ok(()) => true,
            Err(err) => {
                warn(format_args!(
                    "failed to create dump directory {}: {err}",
                    root.display()
                ));
                false
            }
        })
}

fn resolve_root(root: PathBuf) -> Option<PathBuf> {
    let absolute = if root.is_absolute() {
        root
    } else {
        match env::current_dir() {
            Ok(current_dir) => current_dir.join(root),
            Err(err) => {
                warn(format_args!("failed to resolve dump directory: {err}"));
                return None;
            }
        }
    };

    let mut normalized = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            other => normalized.push(other.as_os_str()),
        }
    }
    Some(normalized)
}

fn dump_path(root: &Path, file_name: &str) -> Option<PathBuf> {
    let mut components = Path::new(file_name).components();
    match (components.next(), components.next()) {
        (Some(Component::Normal(name)), None) if !name.is_empty() => Some(root.join(name)),
        _ => {
            warn(format_args!(
                "rejected unsafe debug dump file name {file_name:?}"
            ));
            None
        }
    }
}

fn warn(message: impl Display) {
    eprintln!("warning: plonky2 debug dump: {message}");
}
