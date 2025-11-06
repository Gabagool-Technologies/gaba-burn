use burn::{
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        pool::{MaxPool2d, MaxPool2dConfig, AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
        Dropout, DropoutConfig, Linear, LinearConfig, Relu,
    },
    tensor::{backend::Backend, Tensor},
};

// 1. MicroYOLO-Nano: Ultra-lightweight object detection
#[derive(Module, Debug)]
pub struct MicroYOLONano<B: Backend> {
    // Backbone: Lightweight feature extractor
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    conv3: Conv2d<B>,
    
    // Detection head
    conv_detect: Conv2d<B>,
    
    pool: MaxPool2d,
    activation: Relu,
}

impl<B: Backend> MicroYOLONano<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            // Input: 96x96x3 -> 48x48x16
            conv1: Conv2dConfig::new([3, 16], [3, 3])
                .with_stride([2, 2])
                .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1))
                .init(device),
            
            // 48x48x16 -> 24x24x32
            conv2: Conv2dConfig::new([16, 32], [3, 3])
                .with_stride([2, 2])
                .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1))
                .init(device),
            
            // 24x24x32 -> 12x12x64
            conv3: Conv2dConfig::new([32, 64], [3, 3])
                .with_stride([2, 2])
                .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1))
                .init(device),
            
            // Detection: 12x12x64 -> 12x12x25 (5 boxes * 5 values: x,y,w,h,conf)
            conv_detect: Conv2dConfig::new([64, 25], [1, 1]).init(device),
            
            pool: MaxPool2dConfig::new([2, 2]).init(),
            activation: Relu::new(),
        }
    }
    
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv1.forward(input);
        let x = self.activation.forward(x);
        
        let x = self.conv2.forward(x);
        let x = self.activation.forward(x);
        
        let x = self.conv3.forward(x);
        let x = self.activation.forward(x);
        
        self.conv_detect.forward(x)
    }
}

// 2. EfficientEdge-Lite: Image classification
#[derive(Module, Debug)]
pub struct EfficientEdgeLite<B: Backend> {
    conv1: Conv2d<B>,
    
    // Depthwise separable blocks
    dw_conv1: Conv2d<B>,
    pw_conv1: Conv2d<B>,
    
    dw_conv2: Conv2d<B>,
    pw_conv2: Conv2d<B>,
    
    dw_conv3: Conv2d<B>,
    pw_conv3: Conv2d<B>,
    
    pool: AdaptiveAvgPool2d,
    fc: Linear<B>,
    dropout: Dropout,
    activation: Relu,
}

impl<B: Backend> EfficientEdgeLite<B> {
    pub fn new(num_classes: usize, device: &B::Device) -> Self {
        Self {
            // Input: 128x128x3 -> 64x64x24
            conv1: Conv2dConfig::new([3, 24], [3, 3])
                .with_stride([2, 2])
                .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1))
                .init(device),
            
            // Depthwise separable block 1: 64x64x24 -> 32x32x32
            dw_conv1: Conv2dConfig::new([24, 24], [3, 3])
                .with_stride([2, 2])
                .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1))
                .with_groups(24)
                .init(device),
            pw_conv1: Conv2dConfig::new([24, 32], [1, 1]).init(device),
            
            // Depthwise separable block 2: 32x32x32 -> 16x16x48
            dw_conv2: Conv2dConfig::new([32, 32], [3, 3])
                .with_stride([2, 2])
                .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1))
                .with_groups(32)
                .init(device),
            pw_conv2: Conv2dConfig::new([32, 48], [1, 1]).init(device),
            
            // Depthwise separable block 3: 16x16x48 -> 8x8x64
            dw_conv3: Conv2dConfig::new([48, 48], [3, 3])
                .with_stride([2, 2])
                .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1))
                .with_groups(48)
                .init(device),
            pw_conv3: Conv2dConfig::new([48, 64], [1, 1]).init(device),
            
            pool: AdaptiveAvgPool2dConfig::new([1, 1]).init(),
            fc: LinearConfig::new(64, num_classes).init(device),
            dropout: DropoutConfig::new(0.2).init(),
            activation: Relu::new(),
        }
    }
    
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = self.conv1.forward(input);
        let x = self.activation.forward(x);
        
        // Block 1
        let x = self.dw_conv1.forward(x);
        let x = self.pw_conv1.forward(x);
        let x = self.activation.forward(x);
        
        // Block 2
        let x = self.dw_conv2.forward(x);
        let x = self.pw_conv2.forward(x);
        let x = self.activation.forward(x);
        
        // Block 3
        let x = self.dw_conv3.forward(x);
        let x = self.pw_conv3.forward(x);
        let x = self.activation.forward(x);
        
        // Global pooling and classifier
        let x = self.pool.forward(x);
        let [batch, channels, _, _] = x.dims();
        let x = x.reshape([batch, channels]);
        let x = self.dropout.forward(x);
        self.fc.forward(x)
    }
}

// 3. SegmentMicro: Semantic segmentation
#[derive(Module, Debug)]
pub struct SegmentMicro<B: Backend> {
    // Encoder
    enc_conv1: Conv2d<B>,
    enc_conv2: Conv2d<B>,
    enc_conv3: Conv2d<B>,
    
    // Decoder
    dec_conv1: Conv2d<B>,
    dec_conv2: Conv2d<B>,
    dec_conv3: Conv2d<B>,
    
    pool: MaxPool2d,
    activation: Relu,
}

impl<B: Backend> SegmentMicro<B> {
    pub fn new(num_classes: usize, device: &B::Device) -> Self {
        Self {
            // Encoder: 64x64x3 -> 32x32x16 -> 16x16x32 -> 8x8x64
            enc_conv1: Conv2dConfig::new([3, 16], [3, 3])
                .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1))
                .init(device),
            enc_conv2: Conv2dConfig::new([16, 32], [3, 3])
                .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1))
                .init(device),
            enc_conv3: Conv2dConfig::new([32, 64], [3, 3])
                .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1))
                .init(device),
            
            // Decoder: 8x8x64 -> 16x16x32 -> 32x32x16 -> 64x64xnum_classes
            dec_conv1: Conv2dConfig::new([64, 32], [3, 3])
                .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1))
                .init(device),
            dec_conv2: Conv2dConfig::new([32, 16], [3, 3])
                .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1))
                .init(device),
            dec_conv3: Conv2dConfig::new([16, num_classes], [1, 1]).init(device),
            
            pool: MaxPool2dConfig::new([2, 2]).init(),
            activation: Relu::new(),
        }
    }
    
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        // Encoder
        let x = self.enc_conv1.forward(input);
        let x = self.activation.forward(x);
        let x = self.pool.forward(x);
        
        let x = self.enc_conv2.forward(x);
        let x = self.activation.forward(x);
        let x = self.pool.forward(x);
        
        let x = self.enc_conv3.forward(x);
        let x = self.activation.forward(x);
        let x = self.pool.forward(x);
        
        // Decoder with bilinear upsampling
        let x = self.dec_conv1.forward(x);
        let x = self.activation.forward(x);
        let x = self.upsample(x, 2);
        
        let x = self.dec_conv2.forward(x);
        let x = self.activation.forward(x);
        let x = self.upsample(x, 2);
        
        let x = self.dec_conv3.forward(x);
        self.upsample(x, 2)
    }
    
    fn upsample(&self, x: Tensor<B, 4>, scale: usize) -> Tensor<B, 4> {
        let [_batch, _channels, _height, _width] = x.dims();
        // Simple nearest neighbor upsampling
        let mut result = x;
        for _ in 0..scale.trailing_zeros() {
            let [b, c, h, w] = result.dims();
            result = result.reshape([b, c, h, 1, w, 1])
                .repeat_dim(3, 2)
                .repeat_dim(5, 2)
                .reshape([b, c, h * 2, w * 2]);
        }
        result
    }
}

// 4. FaceDetectNano: Face detection and landmarks
#[derive(Module, Debug)]
pub struct FaceDetectNano<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    conv3: Conv2d<B>,
    conv4: Conv2d<B>,
    
    // Detection head: bounding box
    detect_head: Conv2d<B>,
    // Landmark head: 5 points (x,y) = 10 values
    landmark_head: Conv2d<B>,
    
    pool: MaxPool2d,
    activation: Relu,
}

impl<B: Backend> FaceDetectNano<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            // 80x80x1 -> 40x40x16 -> 20x20x32 -> 10x10x48 -> 5x5x64
            conv1: Conv2dConfig::new([1, 16], [3, 3])
                .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1))
                .init(device),
            conv2: Conv2dConfig::new([16, 32], [3, 3])
                .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1))
                .init(device),
            conv3: Conv2dConfig::new([32, 48], [3, 3])
                .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1))
                .init(device),
            conv4: Conv2dConfig::new([48, 64], [3, 3])
                .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1))
                .init(device),
            
            // Detection: 5 values (x, y, w, h, confidence)
            detect_head: Conv2dConfig::new([64, 5], [1, 1]).init(device),
            // Landmarks: 10 values (5 points * 2 coords)
            landmark_head: Conv2dConfig::new([64, 10], [1, 1]).init(device),
            
            pool: MaxPool2dConfig::new([2, 2]).init(),
            activation: Relu::new(),
        }
    }
    
    pub fn forward(&self, input: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let x = self.conv1.forward(input);
        let x = self.activation.forward(x);
        let x = self.pool.forward(x);
        
        let x = self.conv2.forward(x);
        let x = self.activation.forward(x);
        let x = self.pool.forward(x);
        
        let x = self.conv3.forward(x);
        let x = self.activation.forward(x);
        let x = self.pool.forward(x);
        
        let x = self.conv4.forward(x);
        let x = self.activation.forward(x);
        let x = self.pool.forward(x);
        
        let detections = self.detect_head.forward(x.clone());
        let landmarks = self.landmark_head.forward(x);
        
        (detections, landmarks)
    }
}

// 5. GestureNet-Micro: Hand gesture recognition
#[derive(Module, Debug)]
pub struct GestureNetMicro<B: Backend> {
    // Spatial feature extraction
    spatial_conv1: Conv2d<B>,
    spatial_conv2: Conv2d<B>,
    spatial_conv3: Conv2d<B>,
    
    // Temporal convolution (across frames)
    temporal_conv: Conv2d<B>,
    
    pool: MaxPool2d,
    global_pool: AdaptiveAvgPool2d,
    fc: Linear<B>,
    dropout: Dropout,
    activation: Relu,
}

impl<B: Backend> GestureNetMicro<B> {
    pub fn new(num_gestures: usize, device: &B::Device) -> Self {
        Self {
            // 48x48x1 -> 24x24x16 -> 12x12x32 -> 6x6x48
            spatial_conv1: Conv2dConfig::new([1, 16], [3, 3])
                .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1))
                .init(device),
            spatial_conv2: Conv2dConfig::new([16, 32], [3, 3])
                .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1))
                .init(device),
            spatial_conv3: Conv2dConfig::new([32, 48], [3, 3])
                .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1))
                .init(device),
            
            // Temporal: treat frames as channels
            temporal_conv: Conv2dConfig::new([48, 64], [3, 3])
                .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1))
                .init(device),
            
            pool: MaxPool2dConfig::new([2, 2]).init(),
            global_pool: AdaptiveAvgPool2dConfig::new([1, 1]).init(),
            fc: LinearConfig::new(64, num_gestures).init(device),
            dropout: DropoutConfig::new(0.3).init(),
            activation: Relu::new(),
        }
    }
    
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = self.spatial_conv1.forward(input);
        let x = self.activation.forward(x);
        let x = self.pool.forward(x);
        
        let x = self.spatial_conv2.forward(x);
        let x = self.activation.forward(x);
        let x = self.pool.forward(x);
        
        let x = self.spatial_conv3.forward(x);
        let x = self.activation.forward(x);
        let x = self.pool.forward(x);
        
        let x = self.temporal_conv.forward(x);
        let x = self.activation.forward(x);
        
        let x = self.global_pool.forward(x);
        let [batch, channels, _, _] = x.dims();
        let x = x.reshape([batch, channels]);
        let x = self.dropout.forward(x);
        self.fc.forward(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    
    type TestBackend = NdArray;
    
    #[test]
    fn test_micro_yolo_nano() {
        let device = Default::default();
        let model: MicroYOLONano<TestBackend> = MicroYOLONano::new(&device);
        let input = Tensor::<TestBackend, 4>::zeros([1, 3, 96, 96], &device);
        let output = model.forward(input);
        assert_eq!(output.dims(), [1, 25, 12, 12]);
    }
    
    #[test]
    fn test_efficient_edge_lite() {
        let device = Default::default();
        let model: EfficientEdgeLite<TestBackend> = EfficientEdgeLite::new(10, &device);
        let input = Tensor::<TestBackend, 4>::zeros([1, 3, 128, 128], &device);
        let output = model.forward(input);
        assert_eq!(output.dims(), [1, 10]);
    }
    
    #[test]
    fn test_segment_micro() {
        let device = Default::default();
        let model: SegmentMicro<TestBackend> = SegmentMicro::new(5, &device);
        let input = Tensor::<TestBackend, 4>::zeros([1, 3, 64, 64], &device);
        let output = model.forward(input);
        assert_eq!(output.dims(), [1, 5, 64, 64]);
    }
}
