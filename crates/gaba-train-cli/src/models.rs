//! Simple neural network models using pure Rust with backpropagation

use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use serde::{Serialize, Deserialize};
use std::fs::File;
use std::io::{Write, Read};
use std::path::Path;

/// Traffic speed prediction model (simple 3-layer MLP)
#[derive(Serialize, Deserialize)]
pub struct TrafficModel {
    pub w1: Array2<f32>,  // 22 x 64
    pub b1: Array1<f32>,  // 64
    pub w2: Array2<f32>,  // 64 x 32
    pub b2: Array1<f32>,  // 32
    pub w3: Array2<f32>,  // 32 x 1
    pub b3: Array1<f32>,  // 1
}

/// Intermediate activations for backprop
pub struct TrafficForward {
    pub z1: Array2<f32>,
    pub a1: Array2<f32>,
    pub z2: Array2<f32>,
    pub a2: Array2<f32>,
    pub output: Array2<f32>,
}

/// Gradients for backprop
pub struct TrafficGradients {
    pub dw1: Array2<f32>,
    pub db1: Array1<f32>,
    pub dw2: Array2<f32>,
    pub db2: Array1<f32>,
    pub dw3: Array2<f32>,
    pub db3: Array1<f32>,
}

impl TrafficModel {
    pub fn new() -> Self {
        let mut rng = rand::thread_rng();
        
        // Xavier initialization
        let w1 = Array2::from_shape_fn((22, 64), |_| rng.gen_range(-0.1..0.1));
        let b1 = Array1::zeros(64);
        let w2 = Array2::from_shape_fn((64, 32), |_| rng.gen_range(-0.1..0.1));
        let b2 = Array1::zeros(32);
        let w3 = Array2::from_shape_fn((32, 1), |_| rng.gen_range(-0.1..0.1));
        let b3 = Array1::zeros(1);
        
        Self { w1, b1, w2, b2, w3, b3 }
    }
    
    pub fn forward(&self, x: &Array2<f32>) -> Array1<f32> {
        let fwd = self.forward_cached(x);
        fwd.output.column(0).to_owned()
    }
    
    pub fn forward_cached(&self, x: &Array2<f32>) -> TrafficForward {
        // Layer 1
        let z1 = x.dot(&self.w1) + &self.b1;
        let a1 = z1.mapv(|v| v.max(0.0)); // ReLU
        
        // Layer 2
        let z2 = a1.dot(&self.w2) + &self.b2;
        let a2 = z2.mapv(|v| v.max(0.0)); // ReLU
        
        // Layer 3
        let output = a2.dot(&self.w3) + &self.b3;
        
        TrafficForward { z1, a1, z2, a2, output }
    }
    
    pub fn backward(&self, x: &Array2<f32>, fwd: &TrafficForward, targets: &Array1<f32>) -> TrafficGradients {
        let batch_size = x.nrows() as f32;
        
        // Output gradient (MSE derivative)
        let output_1d = fwd.output.column(0).to_owned();
        let d_output = (&output_1d - targets) * (2.0 / batch_size);
        let d_output_2d = d_output.insert_axis(Axis(1));
        
        // Layer 3 gradients
        let dw3 = fwd.a2.t().dot(&d_output_2d);
        let db3 = d_output_2d.sum_axis(Axis(0));
        
        // Backprop to layer 2
        let d_a2 = d_output_2d.dot(&self.w3.t());
        let d_z2 = &d_a2 * &fwd.z2.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });
        
        // Layer 2 gradients
        let dw2 = fwd.a1.t().dot(&d_z2);
        let db2 = d_z2.sum_axis(Axis(0));
        
        // Backprop to layer 1
        let d_a1 = d_z2.dot(&self.w2.t());
        let d_z1 = &d_a1 * &fwd.z1.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });
        
        // Layer 1 gradients
        let dw1 = x.t().dot(&d_z1);
        let db1 = d_z1.sum_axis(Axis(0));
        
        TrafficGradients { dw1, db1, dw2, db2, dw3, db3 }
    }
    
    pub fn update(&mut self, grads: &TrafficGradients, learning_rate: f32) {
        self.w1 = &self.w1 - &(&grads.dw1 * learning_rate);
        self.b1 = &self.b1 - &(&grads.db1 * learning_rate);
        self.w2 = &self.w2 - &(&grads.dw2 * learning_rate);
        self.b2 = &self.b2 - &(&grads.db2 * learning_rate);
        self.w3 = &self.w3 - &(&grads.dw3 * learning_rate);
        self.b3 = &self.b3 - &(&grads.db3 * learning_rate);
    }
    
    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        let encoded = bincode::serialize(self)?;
        let mut file = File::create(path)?;
        file.write_all(&encoded)?;
        Ok(())
    }
    
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let mut file = File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        let model = bincode::deserialize(&buffer)?;
        Ok(model)
    }
}

/// Route time prediction model (simple 3-layer MLP)
#[derive(Serialize, Deserialize)]
pub struct RouteModel {
    pub w1: Array2<f32>,  // 15 x 64
    pub b1: Array1<f32>,  // 64
    pub w2: Array2<f32>,  // 64 x 32
    pub b2: Array1<f32>,  // 32
    pub w3: Array2<f32>,  // 32 x 1
    pub b3: Array1<f32>,  // 1
}

pub struct RouteForward {
    pub z1: Array2<f32>,
    pub a1: Array2<f32>,
    pub z2: Array2<f32>,
    pub a2: Array2<f32>,
    pub output: Array2<f32>,
}

pub struct RouteGradients {
    pub dw1: Array2<f32>,
    pub db1: Array1<f32>,
    pub dw2: Array2<f32>,
    pub db2: Array1<f32>,
    pub dw3: Array2<f32>,
    pub db3: Array1<f32>,
}

impl RouteModel {
    pub fn new() -> Self {
        let mut rng = rand::thread_rng();
        
        let w1 = Array2::from_shape_fn((15, 64), |_| rng.gen_range(-0.1..0.1));
        let b1 = Array1::zeros(64);
        let w2 = Array2::from_shape_fn((64, 32), |_| rng.gen_range(-0.1..0.1));
        let b2 = Array1::zeros(32);
        let w3 = Array2::from_shape_fn((32, 1), |_| rng.gen_range(-0.1..0.1));
        let b3 = Array1::zeros(1);
        
        Self { w1, b1, w2, b2, w3, b3 }
    }
    
    pub fn forward(&self, x: &Array2<f32>) -> Array1<f32> {
        let fwd = self.forward_cached(x);
        fwd.output.column(0).to_owned()
    }
    
    pub fn forward_cached(&self, x: &Array2<f32>) -> RouteForward {
        let z1 = x.dot(&self.w1) + &self.b1;
        let a1 = z1.mapv(|v| v.max(0.0));
        let z2 = a1.dot(&self.w2) + &self.b2;
        let a2 = z2.mapv(|v| v.max(0.0));
        let output = a2.dot(&self.w3) + &self.b3;
        RouteForward { z1, a1, z2, a2, output }
    }
    
    pub fn backward(&self, x: &Array2<f32>, fwd: &RouteForward, targets: &Array1<f32>) -> RouteGradients {
        let batch_size = x.nrows() as f32;
        let output_1d = fwd.output.column(0).to_owned();
        let d_output = (&output_1d - targets) * (2.0 / batch_size);
        let d_output_2d = d_output.insert_axis(Axis(1));
        
        let dw3 = fwd.a2.t().dot(&d_output_2d);
        let db3 = d_output_2d.sum_axis(Axis(0));
        
        let d_a2 = d_output_2d.dot(&self.w3.t());
        let d_z2 = &d_a2 * &fwd.z2.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });
        let dw2 = fwd.a1.t().dot(&d_z2);
        let db2 = d_z2.sum_axis(Axis(0));
        
        let d_a1 = d_z2.dot(&self.w2.t());
        let d_z1 = &d_a1 * &fwd.z1.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });
        let dw1 = x.t().dot(&d_z1);
        let db1 = d_z1.sum_axis(Axis(0));
        
        RouteGradients { dw1, db1, dw2, db2, dw3, db3 }
    }
    
    pub fn update(&mut self, grads: &RouteGradients, learning_rate: f32) {
        self.w1 = &self.w1 - &(&grads.dw1 * learning_rate);
        self.b1 = &self.b1 - &(&grads.db1 * learning_rate);
        self.w2 = &self.w2 - &(&grads.dw2 * learning_rate);
        self.b2 = &self.b2 - &(&grads.db2 * learning_rate);
        self.w3 = &self.w3 - &(&grads.dw3 * learning_rate);
        self.b3 = &self.b3 - &(&grads.db3 * learning_rate);
    }
    
    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        let encoded = bincode::serialize(self)?;
        let mut file = File::create(path)?;
        file.write_all(&encoded)?;
        Ok(())
    }
    
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let mut file = File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        let model = bincode::deserialize(&buffer)?;
        Ok(model)
    }
}
