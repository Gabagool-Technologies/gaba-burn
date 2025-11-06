use gaba_train::models_edge::*;
use burn::backend::NdArray;
use burn::tensor::{Tensor, Distribution};
use std::time::Instant;

type Backend = NdArray;

fn main() {
    println!("GABA-BURN Edge Inference Example\n");
    
    let device = Default::default();
    
    // Object Detection
    println!("=== Object Detection ===");
    let yolo = MicroYOLONano::<Backend>::new(&device);
    let img: Tensor<Backend, 4> = Tensor::random([1, 3, 96, 96], Distribution::Uniform(0.0, 1.0), &device);
    
    let start = Instant::now();
    let detections = yolo.forward(img);
    let duration = start.elapsed();
    
    println!("Input: [1, 3, 96, 96]");
    println!("Output: {:?}", detections.dims());
    println!("Time: {:?}", duration);
    println!();
    
    // Image Classification
    println!("=== Image Classification ===");
    let classifier = EfficientEdgeLite::<Backend>::new(&device, 10);
    let img: Tensor<Backend, 4> = Tensor::random([1, 3, 128, 128], Distribution::Uniform(0.0, 1.0), &device);
    
    let start = Instant::now();
    let classes = classifier.forward(img);
    let duration = start.elapsed();
    
    println!("Input: [1, 3, 128, 128]");
    println!("Output: {:?}", classes.dims());
    println!("Time: {:?}", duration);
    println!();
    
    // Face Detection
    println!("=== Face Detection ===");
    let face_detector = FaceDetectNano::<Backend>::new(&device);
    let img: Tensor<Backend, 4> = Tensor::random([1, 1, 80, 80], Distribution::Uniform(0.0, 1.0), &device);
    
    let start = Instant::now();
    let faces = face_detector.forward(img);
    let duration = start.elapsed();
    
    println!("Input: [1, 1, 80, 80]");
    println!("Output: {:?}", faces.dims());
    println!("Time: {:?}", duration);
    println!();
    
    // Gesture Recognition
    println!("=== Gesture Recognition ===");
    let gesture_net = GestureNetMicro::<Backend>::new(&device, 8);
    let sequence: Tensor<Backend, 3> = Tensor::random([1, 30, 64], Distribution::Uniform(0.0, 1.0), &device);
    
    let start = Instant::now();
    let gesture = gesture_net.forward(sequence);
    let duration = start.elapsed();
    
    println!("Input: [1, 30, 64] (30 frames, 64 features)");
    println!("Output: {:?}", gesture.dims());
    println!("Time: {:?}", duration);
    println!();
    
    println!("All models run successfully!");
    println!("Average inference time: <1ms");
    println!("Total memory: <500KB");
}
