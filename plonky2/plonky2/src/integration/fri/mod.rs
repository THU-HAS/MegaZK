//! Active GPU FRI services and owning state.
//!
//! Global phase and transcript ordering live exclusively in `GpuProverOrchestrator`.

mod commit;
mod pow;
mod queries;
mod state;

pub(crate) use commit::FriCommitService;
pub(crate) use pow::{FriPowInputs, FriPowService};
pub(crate) use queries::{FriQueryConfig, FriQueryService};
pub(crate) use state::{
    FriCommitBuffers, FriCommitState, FriPowBuffers, FriQueryBuffers, FriQueryGeometry,
    FriQueryState, FriState,
};
