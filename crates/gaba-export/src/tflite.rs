use crate::{ExportError, ModelExporter, Result};
use std::path::Path;

pub struct TfLiteExporter {
    model_name: String,
    subgraphs: Vec<Subgraph>,
    buffers: Vec<Buffer>,
}

pub struct Subgraph {
    pub name: String,
    pub tensors: Vec<Tensor>,
    pub inputs: Vec<u32>,
    pub outputs: Vec<u32>,
    pub operators: Vec<Operator>,
}

pub struct Tensor {
    pub name: String,
    pub shape: Vec<i32>,
    pub tensor_type: TensorType,
    pub buffer_index: u32,
    pub quantization: Option<QuantizationParameters>,
}

pub enum TensorType {
    Float32,
    Float16,
    Int8,
    Int32,
    Uint8,
}

pub struct QuantizationParameters {
    pub scale: Vec<f32>,
    pub zero_point: Vec<i64>,
    pub quantized_dimension: i32,
}

pub struct Operator {
    pub opcode_index: u32,
    pub inputs: Vec<u32>,
    pub outputs: Vec<u32>,
    pub builtin_options: Option<BuiltinOptions>,
}

pub enum BuiltinOptions {
    Conv2D {
        padding: Padding,
        stride_w: i32,
        stride_h: i32,
        fused_activation: ActivationFunction,
    },
    FullyConnected {
        fused_activation: ActivationFunction,
        weights_format: FullyConnectedWeightsFormat,
    },
    Pool2D {
        padding: Padding,
        stride_w: i32,
        stride_h: i32,
        filter_width: i32,
        filter_height: i32,
        fused_activation: ActivationFunction,
    },
}

pub enum Padding {
    Same,
    Valid,
}

pub enum ActivationFunction {
    None,
    Relu,
    Relu6,
    Tanh,
    Sigmoid,
}

pub enum FullyConnectedWeightsFormat {
    Default,
    Shuffled4x16Int8,
}

pub struct Buffer {
    pub data: Vec<u8>,
}

impl TfLiteExporter {
    pub fn new(model_name: impl Into<String>) -> Self {
        Self {
            model_name: model_name.into(),
            subgraphs: Vec::new(),
            buffers: vec![Buffer { data: Vec::new() }],
        }
    }

    pub fn add_subgraph(&mut self, subgraph: Subgraph) {
        self.subgraphs.push(subgraph);
    }

    pub fn add_buffer(&mut self, data: Vec<u8>) -> u32 {
        let index = self.buffers.len() as u32;
        self.buffers.push(Buffer { data });
        index
    }

    #[cfg(feature = "tflite")]
    fn build_flatbuffer(&self) -> Result<Vec<u8>> {
        Err(ExportError::Other(
            "TFLite flatbuffer generation not yet implemented".to_string(),
        ))
    }
}

impl ModelExporter for TfLiteExporter {
    fn export<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        #[cfg(feature = "tflite")]
        {
            let bytes = self.build_flatbuffer()?;
            std::fs::write(path, bytes)?;
            Ok(())
        }
        
        #[cfg(not(feature = "tflite"))]
        {
            let _ = path;
            Err(ExportError::Other(
                "TFLite feature not enabled".to_string(),
            ))
        }
    }

    fn export_bytes(&self) -> Result<Vec<u8>> {
        #[cfg(feature = "tflite")]
        {
            self.build_flatbuffer()
        }
        
        #[cfg(not(feature = "tflite"))]
        {
            Err(ExportError::Other(
                "TFLite feature not enabled".to_string(),
            ))
        }
    }
}

pub fn export_conv2d(
    subgraph: &mut Subgraph,
    exporter: &mut TfLiteExporter,
    name: &str,
    input_tensor: u32,
    weight_data: &[f32],
    bias_data: Option<&[f32]>,
    output_channels: usize,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: Padding,
) -> u32 {
    let weight_buffer = exporter.add_buffer(
        weight_data
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect(),
    );
    
    let weight_tensor = subgraph.tensors.len() as u32;
    subgraph.tensors.push(Tensor {
        name: format!("{}_weight", name),
        shape: vec![
            output_channels as i32,
            kernel_size.0 as i32,
            kernel_size.1 as i32,
            1,
        ],
        tensor_type: TensorType::Float32,
        buffer_index: weight_buffer,
        quantization: None,
    });

    let mut inputs = vec![input_tensor, weight_tensor];

    if let Some(bias) = bias_data {
        let bias_buffer = exporter.add_buffer(
            bias.iter().flat_map(|f| f.to_le_bytes()).collect(),
        );
        
        let bias_tensor = subgraph.tensors.len() as u32;
        subgraph.tensors.push(Tensor {
            name: format!("{}_bias", name),
            shape: vec![output_channels as i32],
            tensor_type: TensorType::Float32,
            buffer_index: bias_buffer,
            quantization: None,
        });
        inputs.push(bias_tensor);
    }

    let output_tensor = subgraph.tensors.len() as u32;
    subgraph.tensors.push(Tensor {
        name: format!("{}_output", name),
        shape: vec![-1, -1, -1, output_channels as i32],
        tensor_type: TensorType::Float32,
        buffer_index: 0,
        quantization: None,
    });

    subgraph.operators.push(Operator {
        opcode_index: 3,
        inputs,
        outputs: vec![output_tensor],
        builtin_options: Some(BuiltinOptions::Conv2D {
            padding,
            stride_w: stride.1 as i32,
            stride_h: stride.0 as i32,
            fused_activation: ActivationFunction::None,
        }),
    });

    output_tensor
}

pub fn export_fully_connected(
    subgraph: &mut Subgraph,
    exporter: &mut TfLiteExporter,
    name: &str,
    input_tensor: u32,
    weight_data: &[f32],
    bias_data: Option<&[f32]>,
    in_features: usize,
    out_features: usize,
) -> u32 {
    let weight_buffer = exporter.add_buffer(
        weight_data
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect(),
    );
    
    let weight_tensor = subgraph.tensors.len() as u32;
    subgraph.tensors.push(Tensor {
        name: format!("{}_weight", name),
        shape: vec![out_features as i32, in_features as i32],
        tensor_type: TensorType::Float32,
        buffer_index: weight_buffer,
        quantization: None,
    });

    let mut inputs = vec![input_tensor, weight_tensor];

    if let Some(bias) = bias_data {
        let bias_buffer = exporter.add_buffer(
            bias.iter().flat_map(|f| f.to_le_bytes()).collect(),
        );
        
        let bias_tensor = subgraph.tensors.len() as u32;
        subgraph.tensors.push(Tensor {
            name: format!("{}_bias", name),
            shape: vec![out_features as i32],
            tensor_type: TensorType::Float32,
            buffer_index: bias_buffer,
            quantization: None,
        });
        inputs.push(bias_tensor);
    }

    let output_tensor = subgraph.tensors.len() as u32;
    subgraph.tensors.push(Tensor {
        name: format!("{}_output", name),
        shape: vec![-1, out_features as i32],
        tensor_type: TensorType::Float32,
        buffer_index: 0,
        quantization: None,
    });

    subgraph.operators.push(Operator {
        opcode_index: 9,
        inputs,
        outputs: vec![output_tensor],
        builtin_options: Some(BuiltinOptions::FullyConnected {
            fused_activation: ActivationFunction::None,
            weights_format: FullyConnectedWeightsFormat::Default,
        }),
    });

    output_tensor
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_exporter() {
        let exporter = TfLiteExporter::new("test_model");
        assert_eq!(exporter.model_name, "test_model");
        assert_eq!(exporter.buffers.len(), 1);
    }

    #[test]
    fn test_add_buffer() {
        let mut exporter = TfLiteExporter::new("test");
        let data = vec![1u8, 2, 3, 4];
        let index = exporter.add_buffer(data.clone());
        assert_eq!(index, 1);
        assert_eq!(exporter.buffers[index as usize].data, data);
    }
}
