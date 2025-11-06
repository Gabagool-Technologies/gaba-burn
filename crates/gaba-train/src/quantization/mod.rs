pub mod gptq;
pub mod awq;
pub mod smoothquant;
pub mod groupwise;
pub mod mixed_precision;

pub use gptq::GptqQuantizer;
pub use awq::AwqQuantizer;
pub use smoothquant::SmoothQuantizer;
pub use groupwise::GroupwiseQuantizer;
pub use mixed_precision::MixedPrecisionQuantizer;

#[derive(Debug, Clone)]
pub struct QuantizationConfig {
    pub enabled: bool,
    pub bits: u8,
    pub method: QuantizationMethod,
}

#[derive(Debug, Clone, Copy)]
pub enum QuantizationMethod {
    Symmetric,
    Asymmetric,
    GPTQ,
    AWQ,
    SmoothQuant,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            bits: 8,
            method: QuantizationMethod::Symmetric,
        }
    }
}
