#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use crate::error::Error;

/// Minimal device abstraction for allocation, copy and kernel execution.
///
/// This is intentionally small for now; implementers can provide concrete
/// buffer types via `Buffer` associated type.
pub trait Device {
    /// Opaque buffer/handle type for device allocations.
    type Buffer;

    /// Allocate a buffer of `size` bytes on the device.
    fn alloc(&self, size: usize) -> Result<Self::Buffer, Error>;

    /// Copy host bytes into the given device buffer.
    fn copy_to_device(&self, dst: &mut Self::Buffer, src: &[u8]) -> Result<(), Error>;

    /// Execute a small named kernel with raw bytes arguments.
    /// Implementations are free to interpret `kernel` and `args` as needed.
    fn run_kernel(&self, kernel: &str, args: &[u8]) -> Result<(), Error>;
}
