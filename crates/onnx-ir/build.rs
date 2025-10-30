fn main() {
    // Generate ONNX protobuf Rust sources into OUT_DIR/onnx-protos using
    // `protobuf-codegen`. This produces files like `onnx.u.pb.rs` which we
    // re-export via a generated `mod.rs` consumed by `src/protos/mod.rs`.
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR must be set");
    let out_path = std::path::Path::new(&out_dir).join("onnx-protos");
    std::fs::create_dir_all(&out_path).expect("create out dir");

    protobuf_codegen::CodeGen::new()
        .includes(["src/protos"].iter())
        .input("onnx.proto")
        .output_dir(&out_path)
        .generate_and_compile()
        .expect("protoc codegen failed");

    // Create a `mod.rs` that includes the generated file and exposes `pub mod onnx`.
    // The generated filename uses the pattern `<proto>.u.pb.rs`.
    let generated_file = out_path.join("onnx.u.pb.rs");
    let mod_rs = out_path.join("mod.rs");

    let content = format!(
        "// Generated wrapper to expose the ONNX proto module.\npub mod onnx {{ include!(\"{}\"); }}\n",
        generated_file.display()
    );

    std::fs::write(&mod_rs, content).expect("failed to write onnx-protos/mod.rs");
}
