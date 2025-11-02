pub use burn_common::device::*;

use alloc::format;
use alloc::string::String;

/// Device ID type
pub type DeviceId = String;

/// The handle device trait allows to get an id for a backend device.
pub trait DeviceOps: Clone + Default + PartialEq + Send + Sync + core::fmt::Debug {
    /// Returns the device id.
    fn id(&self) -> DeviceId {
        format!("{:?}", self)
    }
}
