fn main() {
    #[cfg(feature = "onnx")]
    {
        let mut config = prost_build::Config::new();
        config.type_attribute(".", "#[derive(serde::Serialize, serde::Deserialize)]");
        
        config
            .compile_protos(&["proto/onnx.proto"], &["proto/"])
            .expect("Failed to compile ONNX protobuf");
    }
}
