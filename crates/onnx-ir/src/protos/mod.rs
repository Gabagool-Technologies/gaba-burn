// Include the generated protobuf glue. The codegen produces a `generated.rs`
// which sets up a (private) module and re-exports its contents. Including
// that file here matches the new protobuf v4 codegen layout and ensures
// internal `super::` references resolve correctly.
include!(concat!(env!("OUT_DIR"), "/onnx-protos/generated.rs"));

// Re-export everything produced by the generated module at `crate::protos`.
pub use internal_do_not_use_onnx::*;
