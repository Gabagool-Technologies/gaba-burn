#[cfg(feature = "onnx")]
use prost::Message;

use crate::{ExportError, ModelExporter, Result};
use std::collections::HashMap;
use std::path::Path;

#[cfg(feature = "onnx")]
include!(concat!(env!("OUT_DIR"), "/onnx.rs"));

pub struct OnnxExporter {
    model_name: String,
    producer_name: String,
    producer_version: String,
    opset_version: i64,
    nodes: Vec<OnnxNode>,
    initializers: HashMap<String, TensorData>,
    inputs: Vec<ValueInfo>,
    outputs: Vec<ValueInfo>,
}

pub struct OnnxNode {
    pub name: String,
    pub op_type: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attributes: HashMap<String, AttributeValue>,
}

pub enum AttributeValue {
    Float(f32),
    Int(i64),
    String(String),
    Floats(Vec<f32>),
    Ints(Vec<i64>),
    Tensor(TensorData),
}

pub struct TensorData {
    pub dims: Vec<i64>,
    pub data_type: DataType,
    pub data: Vec<u8>,
}

pub enum DataType {
    Float32,
    Float16,
    Int8,
    Int32,
    Int64,
    Uint8,
}

pub struct ValueInfo {
    pub name: String,
    pub shape: Vec<i64>,
    pub data_type: DataType,
}

impl OnnxExporter {
    pub fn new(model_name: impl Into<String>) -> Self {
        Self {
            model_name: model_name.into(),
            producer_name: "Gaba-Burn".to_string(),
            producer_version: env!("CARGO_PKG_VERSION").to_string(),
            opset_version: 18,
            nodes: Vec::new(),
            initializers: HashMap::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }

    pub fn add_node(&mut self, node: OnnxNode) {
        self.nodes.push(node);
    }

    pub fn add_initializer(&mut self, name: String, tensor: TensorData) {
        self.initializers.insert(name, tensor);
    }

    pub fn add_input(&mut self, input: ValueInfo) {
        self.inputs.push(input);
    }

    pub fn add_output(&mut self, output: ValueInfo) {
        self.outputs.push(output);
    }

    #[cfg(feature = "onnx")]
    fn build_proto(&self) -> Result<ModelProto> {
        let mut graph = GraphProto {
            name: self.model_name.clone(),
            node: Vec::new(),
            initializer: Vec::new(),
            input: Vec::new(),
            output: Vec::new(),
            value_info: Vec::new(),
        };

        for node in &self.nodes {
            let mut proto_node = NodeProto {
                name: node.name.clone(),
                op_type: node.op_type.clone(),
                input: node.inputs.clone(),
                output: node.outputs.clone(),
                attribute: Vec::new(),
            };

            for (key, value) in &node.attributes {
                let mut attr = AttributeProto {
                    name: key.clone(),
                    r#type: 0,
                    f: 0.0,
                    i: 0,
                    s: Vec::new(),
                    t: None,
                    floats: Vec::new(),
                    ints: Vec::new(),
                    strings: Vec::new(),
                };

                match value {
                    AttributeValue::Float(f) => {
                        attr.r#type = 1;
                        attr.f = *f;
                    }
                    AttributeValue::Int(i) => {
                        attr.r#type = 2;
                        attr.i = *i;
                    }
                    AttributeValue::String(s) => {
                        attr.r#type = 3;
                        attr.s = s.as_bytes().to_vec();
                    }
                    AttributeValue::Floats(floats) => {
                        attr.r#type = 6;
                        attr.floats = floats.clone();
                    }
                    AttributeValue::Ints(ints) => {
                        attr.r#type = 7;
                        attr.ints = ints.clone();
                    }
                    AttributeValue::Tensor(_) => {
                        attr.r#type = 4;
                    }
                }

                proto_node.attribute.push(attr);
            }

            graph.node.push(proto_node);
        }

        for (name, tensor) in &self.initializers {
            graph.initializer.push(TensorProto {
                dims: tensor.dims.clone(),
                data_type: Self::data_type_to_proto(&tensor.data_type),
                raw_data: tensor.data.clone(),
                float_data: Vec::new(),
                int32_data: Vec::new(),
                int64_data: Vec::new(),
                name: name.clone(),
            });
        }

        for input in &self.inputs {
            graph.input.push(Self::value_info_to_proto(input));
        }

        for output in &self.outputs {
            graph.output.push(Self::value_info_to_proto(output));
        }

        Ok(ModelProto {
            ir_version: 8,
            opset_import: vec!["ai.onnx:18".to_string()],
            producer_name: self.producer_name.clone(),
            producer_version: self.producer_version.clone(),
            domain: String::new(),
            model_version: 1,
            graph: Some(graph),
        })
    }

    #[cfg(feature = "onnx")]
    fn data_type_to_proto(dt: &DataType) -> i32 {
        match dt {
            DataType::Float32 => 1,
            DataType::Float16 => 10,
            DataType::Int8 => 3,
            DataType::Int32 => 6,
            DataType::Int64 => 7,
            DataType::Uint8 => 2,
        }
    }

    #[cfg(feature = "onnx")]
    fn value_info_to_proto(info: &ValueInfo) -> ValueInfoProto {
        let shape = TensorShapeProto {
            dim: info
                .shape
                .iter()
                .map(|&d| tensor_shape_proto::Dimension {
                    value: Some(tensor_shape_proto::dimension::Value::DimValue(d)),
                })
                .collect(),
        };

        let tensor_type = type_proto::Tensor {
            elem_type: Self::data_type_to_proto(&info.data_type),
            shape: Some(shape),
        };

        ValueInfoProto {
            name: info.name.clone(),
            r#type: Some(TypeProto {
                value: Some(type_proto::Value::TensorType(tensor_type)),
            }),
        }
    }
}

#[cfg(feature = "onnx")]
impl ModelExporter for OnnxExporter {
    fn export<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let proto = self.build_proto()?;
        let bytes = proto.encode_to_vec();
        std::fs::write(path, bytes)?;
        Ok(())
    }

    fn export_bytes(&self) -> Result<Vec<u8>> {
        let proto = self.build_proto()?;
        Ok(proto.encode_to_vec())
    }
}

#[cfg(not(feature = "onnx"))]
impl ModelExporter for OnnxExporter {
    fn export<P: AsRef<Path>>(&self, _path: P) -> Result<()> {
        Err(ExportError::Other(
            "ONNX feature not enabled".to_string(),
        ))
    }

    fn export_bytes(&self) -> Result<Vec<u8>> {
        Err(ExportError::Other(
            "ONNX feature not enabled".to_string(),
        ))
    }
}

pub fn export_linear_layer(
    exporter: &mut OnnxExporter,
    name: &str,
    input_name: &str,
    output_name: &str,
    weight: &[f32],
    bias: Option<&[f32]>,
    in_features: usize,
    out_features: usize,
) {
    let weight_name = format!("{}_weight", name);
    exporter.add_initializer(
        weight_name.clone(),
        TensorData {
            dims: vec![out_features as i64, in_features as i64],
            data_type: DataType::Float32,
            data: weight
                .iter()
                .flat_map(|f| f.to_le_bytes())
                .collect(),
        },
    );

    let mut inputs = vec![input_name.to_string(), weight_name];
    
    if let Some(bias_data) = bias {
        let bias_name = format!("{}_bias", name);
        exporter.add_initializer(
            bias_name.clone(),
            TensorData {
                dims: vec![out_features as i64],
                data_type: DataType::Float32,
                data: bias_data
                    .iter()
                    .flat_map(|f| f.to_le_bytes())
                    .collect(),
            },
        );
        inputs.push(bias_name);
    }

    exporter.add_node(OnnxNode {
        name: name.to_string(),
        op_type: "Gemm".to_string(),
        inputs,
        outputs: vec![output_name.to_string()],
        attributes: HashMap::from([
            ("alpha".to_string(), AttributeValue::Float(1.0)),
            ("beta".to_string(), AttributeValue::Float(1.0)),
            ("transB".to_string(), AttributeValue::Int(1)),
        ]),
    });
}

pub fn export_conv2d_layer(
    exporter: &mut OnnxExporter,
    name: &str,
    input_name: &str,
    output_name: &str,
    weight: &[f32],
    bias: Option<&[f32]>,
    in_channels: usize,
    out_channels: usize,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
) {
    let weight_name = format!("{}_weight", name);
    exporter.add_initializer(
        weight_name.clone(),
        TensorData {
            dims: vec![
                out_channels as i64,
                in_channels as i64,
                kernel_size.0 as i64,
                kernel_size.1 as i64,
            ],
            data_type: DataType::Float32,
            data: weight
                .iter()
                .flat_map(|f| f.to_le_bytes())
                .collect(),
        },
    );

    let mut inputs = vec![input_name.to_string(), weight_name];
    
    if let Some(bias_data) = bias {
        let bias_name = format!("{}_bias", name);
        exporter.add_initializer(
            bias_name.clone(),
            TensorData {
                dims: vec![out_channels as i64],
                data_type: DataType::Float32,
                data: bias_data
                    .iter()
                    .flat_map(|f| f.to_le_bytes())
                    .collect(),
            },
        );
        inputs.push(bias_name);
    }

    exporter.add_node(OnnxNode {
        name: name.to_string(),
        op_type: "Conv".to_string(),
        inputs,
        outputs: vec![output_name.to_string()],
        attributes: HashMap::from([
            (
                "kernel_shape".to_string(),
                AttributeValue::Ints(vec![kernel_size.0 as i64, kernel_size.1 as i64]),
            ),
            (
                "strides".to_string(),
                AttributeValue::Ints(vec![stride.0 as i64, stride.1 as i64]),
            ),
            (
                "pads".to_string(),
                AttributeValue::Ints(vec![
                    padding.0 as i64,
                    padding.1 as i64,
                    padding.0 as i64,
                    padding.1 as i64,
                ]),
            ),
        ]),
    });
}

pub fn export_relu(
    exporter: &mut OnnxExporter,
    name: &str,
    input_name: &str,
    output_name: &str,
) {
    exporter.add_node(OnnxNode {
        name: name.to_string(),
        op_type: "Relu".to_string(),
        inputs: vec![input_name.to_string()],
        outputs: vec![output_name.to_string()],
        attributes: HashMap::new(),
    });
}

pub fn export_maxpool2d(
    exporter: &mut OnnxExporter,
    name: &str,
    input_name: &str,
    output_name: &str,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
) {
    exporter.add_node(OnnxNode {
        name: name.to_string(),
        op_type: "MaxPool".to_string(),
        inputs: vec![input_name.to_string()],
        outputs: vec![output_name.to_string()],
        attributes: HashMap::from([
            (
                "kernel_shape".to_string(),
                AttributeValue::Ints(vec![kernel_size.0 as i64, kernel_size.1 as i64]),
            ),
            (
                "strides".to_string(),
                AttributeValue::Ints(vec![stride.0 as i64, stride.1 as i64]),
            ),
            (
                "pads".to_string(),
                AttributeValue::Ints(vec![
                    padding.0 as i64,
                    padding.1 as i64,
                    padding.0 as i64,
                    padding.1 as i64,
                ]),
            ),
        ]),
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_exporter() {
        let exporter = OnnxExporter::new("test_model");
        assert_eq!(exporter.model_name, "test_model");
        assert_eq!(exporter.producer_name, "Gaba-Burn");
    }

    #[test]
    fn test_add_node() {
        let mut exporter = OnnxExporter::new("test");
        export_relu(&mut exporter, "relu1", "input", "output");
        assert_eq!(exporter.nodes.len(), 1);
        assert_eq!(exporter.nodes[0].op_type, "Relu");
    }

    #[test]
    #[cfg(feature = "onnx")]
    fn test_export_simple_model() {
        let mut exporter = OnnxExporter::new("simple_model");
        
        exporter.add_input(ValueInfo {
            name: "input".to_string(),
            shape: vec![1, 10],
            data_type: DataType::Float32,
        });

        let weight = vec![0.1f32; 100];
        let bias = vec![0.0f32; 10];
        export_linear_layer(
            &mut exporter,
            "fc1",
            "input",
            "output",
            &weight,
            Some(&bias),
            10,
            10,
        );

        exporter.add_output(ValueInfo {
            name: "output".to_string(),
            shape: vec![1, 10],
            data_type: DataType::Float32,
        });

        let bytes = exporter.export_bytes();
        assert!(bytes.is_ok());
        assert!(!bytes.unwrap().is_empty());
    }
}
