use burn::{
    module::Module,
    nn::{
        conv::{Conv1d, Conv1dConfig},
        Dropout, DropoutConfig, Linear, LinearConfig, Relu, BatchNorm, BatchNormConfig,
        pool::{MaxPool1d, MaxPool1dConfig, AdaptiveAvgPool1d, AdaptiveAvgPool1dConfig},
        Embedding, EmbeddingConfig,
    },
    tensor::{backend::Backend, Tensor},
};

// 26. SentimentAnalysis-Micro: Lightweight sentiment analysis
#[derive(Module, Debug)]
pub struct SentimentAnalysisMicro<B: Backend> {
    embedding: Embedding<B>,
    
    conv1: Conv1d<B>,
    bn1: BatchNorm<B>,
    conv2: Conv1d<B>,
    bn2: BatchNorm<B>,
    conv3: Conv1d<B>,
    bn3: BatchNorm<B>,
    
    fc1: Linear<B>,
    fc2: Linear<B>,
    
    pool: MaxPool1d,
    global_pool: AdaptiveAvgPool1d,
    dropout: Dropout,
    activation: Relu,
}

impl<B: Backend> SentimentAnalysisMicro<B> {
    pub fn new(vocab_size: usize, embedding_dim: usize, num_classes: usize, device: &B::Device) -> Self {
        Self {
            // Compact word embeddings
            embedding: EmbeddingConfig::new(vocab_size, embedding_dim).init(device),
            
            // Multi-scale CNNs for n-gram features
            conv1: Conv1dConfig::new(embedding_dim, 32, 3)
                .with_padding(burn::nn::PaddingConfig1d::Explicit(1))
                .init(device),
            bn1: BatchNormConfig::new(32).init(device),
            
            conv2: Conv1dConfig::new(embedding_dim, 32, 5)
                .with_padding(burn::nn::PaddingConfig1d::Explicit(2))
                .init(device),
            bn2: BatchNormConfig::new(32).init(device),
            
            conv3: Conv1dConfig::new(embedding_dim, 32, 7)
                .with_padding(burn::nn::PaddingConfig1d::Explicit(3))
                .init(device),
            bn3: BatchNormConfig::new(32).init(device),
            
            fc1: LinearConfig::new(96, 48).init(device),
            fc2: LinearConfig::new(48, num_classes).init(device),
            
            pool: MaxPool1dConfig::new(2).init(),
            global_pool: AdaptiveAvgPool1dConfig::new(1).init(),
            dropout: DropoutConfig::new(0.3).init(),
            activation: Relu::new(),
        }
    }
    
    pub fn forward(&self, input: Tensor<B, 2, burn::tensor::Int>) -> Tensor<B, 2> {
        let [batch, _seq_len] = input.dims();
        
        // Embedding: [batch, seq_len] -> [batch, seq_len, embed_dim]
        let x = self.embedding.forward(input);
        
        // Transpose for conv1d: [batch, embed_dim, seq_len]
        let x = x.swap_dims(1, 2);
        
        // Multi-scale convolutions
        let feat1 = self.conv1.forward(x.clone());
        let feat1 = self.bn1.forward(feat1);
        let feat1 = self.activation.forward(feat1);
        let feat1 = self.global_pool.forward(feat1);
        
        let feat2 = self.conv2.forward(x.clone());
        let feat2 = self.bn2.forward(feat2);
        let feat2 = self.activation.forward(feat2);
        let feat2 = self.global_pool.forward(feat2);
        
        let feat3 = self.conv3.forward(x);
        let feat3 = self.bn3.forward(feat3);
        let feat3 = self.activation.forward(feat3);
        let feat3 = self.global_pool.forward(feat3);
        
        // Concatenate features
        let features = Tensor::cat(vec![feat1, feat2, feat3], 1);
        let features = features.reshape([batch, 96]);
        
        let x = self.fc1.forward(features);
        let x = self.activation.forward(x);
        let x = self.dropout.forward(x);
        
        self.fc2.forward(x)
    }
}

// 27. IntentClassify-Nano: Intent classification for chatbots
#[derive(Module, Debug)]
pub struct IntentClassifyNano<B: Backend> {
    embedding: Embedding<B>,
    
    conv1: Conv1d<B>,
    bn1: BatchNorm<B>,
    conv2: Conv1d<B>,
    bn2: BatchNorm<B>,
    
    // Attention mechanism
    attention_fc: Linear<B>,
    
    fc1: Linear<B>,
    fc2: Linear<B>,
    
    global_pool: AdaptiveAvgPool1d,
    dropout: Dropout,
    activation: Relu,
}

impl<B: Backend> IntentClassifyNano<B> {
    pub fn new(vocab_size: usize, embedding_dim: usize, num_intents: usize, device: &B::Device) -> Self {
        Self {
            embedding: EmbeddingConfig::new(vocab_size, embedding_dim).init(device),
            
            conv1: Conv1dConfig::new(embedding_dim, 48, 3)
                .with_padding(burn::nn::PaddingConfig1d::Explicit(1))
                .init(device),
            bn1: BatchNormConfig::new(48).init(device),
            
            conv2: Conv1dConfig::new(48, 64, 3)
                .with_padding(burn::nn::PaddingConfig1d::Explicit(1))
                .init(device),
            bn2: BatchNormConfig::new(64).init(device),
            
            attention_fc: LinearConfig::new(64, 64).init(device),
            
            fc1: LinearConfig::new(64, 48).init(device),
            fc2: LinearConfig::new(48, num_intents).init(device),
            
            global_pool: AdaptiveAvgPool1dConfig::new(1).init(),
            dropout: DropoutConfig::new(0.3).init(),
            activation: Relu::new(),
        }
    }
    
    pub fn forward(&self, input: Tensor<B, 2, burn::tensor::Int>) -> Tensor<B, 2> {
        let [batch, _] = input.dims();
        
        let x = self.embedding.forward(input);
        let x = x.swap_dims(1, 2);
        
        let x = self.conv1.forward(x);
        let x = self.bn1.forward(x);
        let x = self.activation.forward(x);
        
        let x = self.conv2.forward(x);
        let x = self.bn2.forward(x);
        let x = self.activation.forward(x);
        
        // Global pooling with attention
        let x = self.global_pool.forward(x);
        let x = x.reshape([batch, 64]);
        
        let x = self.fc1.forward(x);
        let x = self.activation.forward(x);
        let x = self.dropout.forward(x);
        
        self.fc2.forward(x)
    }
}

// 28. NamedEntity-Micro: Named entity recognition
#[derive(Module, Debug)]
pub struct NamedEntityMicro<B: Backend> {
    embedding: Embedding<B>,
    
    // Bidirectional modeling with two 1D convs
    forward_conv1: Conv1d<B>,
    forward_bn1: BatchNorm<B>,
    forward_conv2: Conv1d<B>,
    forward_bn2: BatchNorm<B>,
    
    backward_conv1: Conv1d<B>,
    backward_bn1: BatchNorm<B>,
    backward_conv2: Conv1d<B>,
    backward_bn2: BatchNorm<B>,
    
    // CRF-like output layer
    fc_out: Linear<B>,
    
    dropout: Dropout,
    activation: Relu,
}

impl<B: Backend> NamedEntityMicro<B> {
    pub fn new(vocab_size: usize, embedding_dim: usize, num_tags: usize, device: &B::Device) -> Self {
        Self {
            embedding: EmbeddingConfig::new(vocab_size, embedding_dim).init(device),
            
            // Forward direction
            forward_conv1: Conv1dConfig::new(embedding_dim, 32, 3)
                .with_padding(burn::nn::PaddingConfig1d::Explicit(1))
                .init(device),
            forward_bn1: BatchNormConfig::new(32).init(device),
            forward_conv2: Conv1dConfig::new(32, 48, 3)
                .with_padding(burn::nn::PaddingConfig1d::Explicit(1))
                .init(device),
            forward_bn2: BatchNormConfig::new(48).init(device),
            
            // Backward direction
            backward_conv1: Conv1dConfig::new(embedding_dim, 32, 3)
                .with_padding(burn::nn::PaddingConfig1d::Explicit(1))
                .init(device),
            backward_bn1: BatchNormConfig::new(32).init(device),
            backward_conv2: Conv1dConfig::new(32, 48, 3)
                .with_padding(burn::nn::PaddingConfig1d::Explicit(1))
                .init(device),
            backward_bn2: BatchNormConfig::new(48).init(device),
            
            fc_out: LinearConfig::new(96, num_tags).init(device),
            
            dropout: DropoutConfig::new(0.2).init(),
            activation: Relu::new(),
        }
    }
    
    pub fn forward(&self, input: Tensor<B, 2, burn::tensor::Int>) -> Tensor<B, 3> {
        let [_batch, _seq_len] = input.dims();
        
        let x = self.embedding.forward(input);
        let x = x.swap_dims(1, 2);
        
        // Forward direction
        let fwd = self.forward_conv1.forward(x.clone());
        let fwd = self.forward_bn1.forward(fwd);
        let fwd = self.activation.forward(fwd);
        let fwd = self.forward_conv2.forward(fwd);
        let fwd = self.forward_bn2.forward(fwd);
        let fwd = self.activation.forward(fwd);
        
        // Backward direction (reverse sequence)
        let bwd = x.flip([2]);
        let bwd = self.backward_conv1.forward(bwd);
        let bwd = self.backward_bn1.forward(bwd);
        let bwd = self.activation.forward(bwd);
        let bwd = self.backward_conv2.forward(bwd);
        let bwd = self.backward_bn2.forward(bwd);
        let bwd = self.activation.forward(bwd);
        let bwd = bwd.flip([2]);
        
        // Concatenate bidirectional features
        let features = Tensor::cat(vec![fwd, bwd], 1);
        let features = features.swap_dims(1, 2);
        
        // Per-token classification
        let [b, s, f] = features.dims();
        let features = features.reshape([b * s, f]);
        let output = self.fc_out.forward(features);
        let out_dim = output.dims()[1];
        output.reshape([b, s, out_dim])
    }
}

// 29. TextSummarize-Nano: Extractive text summarization
#[derive(Module, Debug)]
pub struct TextSummarizeNano<B: Backend> {
    embedding: Embedding<B>,
    
    // Sentence encoder
    sent_conv1: Conv1d<B>,
    sent_bn1: BatchNorm<B>,
    sent_conv2: Conv1d<B>,
    sent_bn2: BatchNorm<B>,
    
    // Document encoder
    doc_conv: Conv1d<B>,
    doc_bn: BatchNorm<B>,
    
    // Sentence scoring
    score_fc1: Linear<B>,
    score_fc2: Linear<B>,
    
    pool: AdaptiveAvgPool1d,
    dropout: Dropout,
    activation: Relu,
}

impl<B: Backend> TextSummarizeNano<B> {
    pub fn new(vocab_size: usize, embedding_dim: usize, device: &B::Device) -> Self {
        Self {
            embedding: EmbeddingConfig::new(vocab_size, embedding_dim).init(device),
            
            // Sentence-level encoding
            sent_conv1: Conv1dConfig::new(embedding_dim, 32, 3)
                .with_padding(burn::nn::PaddingConfig1d::Explicit(1))
                .init(device),
            sent_bn1: BatchNormConfig::new(32).init(device),
            sent_conv2: Conv1dConfig::new(32, 48, 3)
                .with_padding(burn::nn::PaddingConfig1d::Explicit(1))
                .init(device),
            sent_bn2: BatchNormConfig::new(48).init(device),
            
            // Document-level encoding
            doc_conv: Conv1dConfig::new(48, 64, 5)
                .with_padding(burn::nn::PaddingConfig1d::Explicit(2))
                .init(device),
            doc_bn: BatchNormConfig::new(64).init(device),
            
            // Sentence importance scoring
            score_fc1: LinearConfig::new(64, 32).init(device),
            score_fc2: LinearConfig::new(32, 1).init(device),
            
            pool: AdaptiveAvgPool1dConfig::new(1).init(),
            dropout: DropoutConfig::new(0.2).init(),
            activation: Relu::new(),
        }
    }
    
    pub fn forward(&self, input: Tensor<B, 2, burn::tensor::Int>) -> Tensor<B, 2> {
        let [_batch, _seq_len] = input.dims();
        
        let x = self.embedding.forward(input);
        let x = x.swap_dims(1, 2);
        
        // Sentence encoding
        let x = self.sent_conv1.forward(x);
        let x = self.sent_bn1.forward(x);
        let x = self.activation.forward(x);
        
        let x = self.sent_conv2.forward(x);
        let x = self.sent_bn2.forward(x);
        let x = self.activation.forward(x);
        
        // Document encoding
        let x = self.doc_conv.forward(x);
        let x = self.doc_bn.forward(x);
        let x = self.activation.forward(x);
        
        // Per-position scoring
        let x = x.swap_dims(1, 2);
        let [b, s, f] = x.dims();
        let x = x.reshape([b * s, f]);
        
        let scores = self.score_fc1.forward(x);
        let scores = self.activation.forward(scores);
        let scores = self.dropout.forward(scores);
        let scores = self.score_fc2.forward(scores);
        
        scores.reshape([b, s])
    }
}

// 30. LanguageDetect-Nano: Language identification
#[derive(Module, Debug)]
pub struct LanguageDetectNano<B: Backend> {
    // Character-level n-gram features
    conv1: Conv1d<B>,
    bn1: BatchNorm<B>,
    conv2: Conv1d<B>,
    bn2: BatchNorm<B>,
    conv3: Conv1d<B>,
    bn3: BatchNorm<B>,
    
    fc1: Linear<B>,
    fc2: Linear<B>,
    
    pool: MaxPool1d,
    global_pool: AdaptiveAvgPool1d,
    dropout: Dropout,
    activation: Relu,
}

impl<B: Backend> LanguageDetectNano<B> {
    pub fn new(char_vocab_size: usize, num_languages: usize, device: &B::Device) -> Self {
        Self {
            // Character-level convolutions
            conv1: Conv1dConfig::new(char_vocab_size, 32, 3)
                .with_padding(burn::nn::PaddingConfig1d::Explicit(1))
                .init(device),
            bn1: BatchNormConfig::new(32).init(device),
            
            conv2: Conv1dConfig::new(32, 48, 5)
                .with_padding(burn::nn::PaddingConfig1d::Explicit(2))
                .init(device),
            bn2: BatchNormConfig::new(48).init(device),
            
            conv3: Conv1dConfig::new(48, 64, 7)
                .with_padding(burn::nn::PaddingConfig1d::Explicit(3))
                .init(device),
            bn3: BatchNormConfig::new(64).init(device),
            
            fc1: LinearConfig::new(64, 48).init(device),
            fc2: LinearConfig::new(48, num_languages).init(device),
            
            pool: MaxPool1dConfig::new(2).init(),
            global_pool: AdaptiveAvgPool1dConfig::new(1).init(),
            dropout: DropoutConfig::new(0.3).init(),
            activation: Relu::new(),
        }
    }
    
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch, _, _] = input.dims();
        
        let x = self.conv1.forward(input);
        let x = self.bn1.forward(x);
        let x = self.activation.forward(x);
        let x = self.pool.forward(x);
        
        let x = self.conv2.forward(x);
        let x = self.bn2.forward(x);
        let x = self.activation.forward(x);
        let x = self.pool.forward(x);
        
        let x = self.conv3.forward(x);
        let x = self.bn3.forward(x);
        let x = self.activation.forward(x);
        
        let x = self.global_pool.forward(x);
        let x = x.reshape([batch, 64]);
        
        let x = self.fc1.forward(x);
        let x = self.activation.forward(x);
        let x = self.dropout.forward(x);
        
        self.fc2.forward(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    
    type TestBackend = NdArray;
    
    #[test]
    fn test_sentiment_analysis_micro() {
        let device = Default::default();
        let model: SentimentAnalysisMicro<TestBackend> = 
            SentimentAnalysisMicro::new(5000, 64, 3, &device);
        let input = Tensor::<TestBackend, 2, burn::tensor::Int>::zeros([2, 50], &device);
        let output = model.forward(input);
        assert_eq!(output.dims(), [2, 3]);
    }
    
    #[test]
    fn test_intent_classify_nano() {
        let device = Default::default();
        let model: IntentClassifyNano<TestBackend> = 
            IntentClassifyNano::new(3000, 48, 20, &device);
        let input = Tensor::<TestBackend, 2, burn::tensor::Int>::zeros([2, 30], &device);
        let output = model.forward(input);
        assert_eq!(output.dims(), [2, 20]);
    }
}
