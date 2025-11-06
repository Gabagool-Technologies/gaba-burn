use crate::{Result, ServeError};
use ndarray::Array2;
use std::sync::Arc;
use tokio::sync::mpsc;

pub struct StreamConfig {
    pub buffer_size: usize,
    pub enable_backpressure: bool,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            buffer_size: 100,
            enable_backpressure: true,
        }
    }
}

pub struct StreamingInference {
    config: StreamConfig,
}

pub struct TokenStream {
    receiver: mpsc::Receiver<Token>,
}

pub struct Token {
    pub id: u32,
    pub text: String,
    pub logprob: f32,
}

impl StreamingInference {
    pub fn new(config: StreamConfig) -> Self {
        Self { config }
    }

    pub fn create_stream(&self) -> (mpsc::Sender<Token>, TokenStream) {
        let (tx, rx) = mpsc::channel(self.config.buffer_size);
        (tx, TokenStream { receiver: rx })
    }

    pub async fn generate_stream(
        &self,
        _input: Array2<f32>,
        sender: mpsc::Sender<Token>,
    ) -> Result<()> {
        for i in 0..10 {
            let token = Token {
                id: i,
                text: format!("token_{}", i),
                logprob: -0.1,
            };
            
            if sender.send(token).await.is_err() {
                return Err(ServeError::Other("Stream closed".to_string()));
            }
        }
        
        Ok(())
    }
}

impl TokenStream {
    pub async fn next(&mut self) -> Option<Token> {
        self.receiver.recv().await
    }

    pub fn try_next(&mut self) -> Option<Token> {
        self.receiver.try_recv().ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    #[tokio::test]
    async fn test_streaming_creation() {
        let config = StreamConfig::default();
        let streaming = StreamingInference::new(config);
        let (_tx, _stream) = streaming.create_stream();
    }

    #[tokio::test]
    async fn test_token_stream() {
        let config = StreamConfig::default();
        let streaming = StreamingInference::new(config);
        let (tx, mut stream) = streaming.create_stream();
        
        tokio::spawn(async move {
            for i in 0..5 {
                let token = Token {
                    id: i,
                    text: format!("token_{}", i),
                    logprob: -0.1,
                };
                tx.send(token).await.unwrap();
            }
        });
        
        let mut count = 0;
        while let Some(_token) = stream.next().await {
            count += 1;
            if count >= 5 {
                break;
            }
        }
        
        assert_eq!(count, 5);
    }

    #[tokio::test]
    async fn test_generate_stream() {
        let config = StreamConfig::default();
        let streaming = StreamingInference::new(config);
        let (tx, mut stream) = streaming.create_stream();
        
        let input = Array::from_shape_vec((1, 10), vec![0.1; 10]).unwrap();
        
        tokio::spawn(async move {
            streaming.generate_stream(input, tx).await.unwrap();
        });
        
        let mut count = 0;
        while let Some(_token) = stream.next().await {
            count += 1;
        }
        
        assert_eq!(count, 10);
    }
}
