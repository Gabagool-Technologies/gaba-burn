use burn::{
    module::Module,
    nn::{Dropout, DropoutConfig, Linear, LinearConfig, Relu},
    tensor::{backend::Backend, Tensor},
};

#[derive(Module, Debug)]
pub struct SensorModel<B: Backend> {
    fc1: Linear<B>,
    dropout1: Dropout,
    fc2: Linear<B>,
    dropout2: Dropout,
    fc3: Linear<B>,
    activation: Relu,
}

impl<B: Backend> SensorModel<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            fc1: LinearConfig::new(5, 64).init(device),
            dropout1: DropoutConfig::new(0.3).init(),
            fc2: LinearConfig::new(64, 32).init(device),
            dropout2: DropoutConfig::new(0.3).init(),
            fc3: LinearConfig::new(32, 1).init(device),
            activation: Relu::new(),
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.fc1.forward(input);
        let x = self.activation.forward(x);
        let x = self.dropout1.forward(x);
        let x = self.fc2.forward(x);
        let x = self.activation.forward(x);
        let x = self.dropout2.forward(x);
        self.fc3.forward(x)
    }
}

#[derive(Module, Debug)]
pub struct AnomalyModel<B: Backend> {
    fc1: Linear<B>,
    dropout1: Dropout,
    fc2: Linear<B>,
    dropout2: Dropout,
    fc3: Linear<B>,
    activation: Relu,
}

impl<B: Backend> AnomalyModel<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            fc1: LinearConfig::new(5, 32).init(device),
            dropout1: DropoutConfig::new(0.2).init(),
            fc2: LinearConfig::new(32, 16).init(device),
            dropout2: DropoutConfig::new(0.2).init(),
            fc3: LinearConfig::new(16, 1).init(device),
            activation: Relu::new(),
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.fc1.forward(input);
        let x = self.activation.forward(x);
        let x = self.dropout1.forward(x);
        let x = self.fc2.forward(x);
        let x = self.activation.forward(x);
        let x = self.dropout2.forward(x);
        self.fc3.forward(x)
    }
}
