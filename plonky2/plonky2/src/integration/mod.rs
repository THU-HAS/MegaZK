//! GPU integration module wiring.
//!
//! This file declares module visibility and minimal re-exports only; phase algorithms and buffer
//! ownership belong to their focused modules.

pub mod ntt;
pub mod poseidon;
pub mod utils;
pub mod quotient;
pub mod fri;
pub mod top;

pub(crate) mod commitments;
pub(crate) mod final_poly;
mod layout;
pub(crate) mod openings;
pub(crate) mod orchestrator;
mod overlay;
mod partial_products;
pub(crate) mod proof_assembly;
mod public_inputs;
mod setup;
pub(crate) mod transcript;

pub use layout::{GpuPolynomialSegment, PolyLayout};
pub(crate) use layout::PolySegment;
mod witgen;
