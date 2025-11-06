use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FusionOp {
    MatMul,
    Add,
    ReLU,
    BatchNorm,
    Conv2D,
    MaxPool,
    Softmax,
    LayerNorm,
    GELU,
    Dropout,
}

#[derive(Debug, Clone)]
pub struct FusionPattern {
    pub ops: Vec<FusionOp>,
    pub name: String,
    pub speedup_factor: f32,
}

pub struct FusionEngine {
    patterns: Vec<FusionPattern>,
    fusion_cache: HashMap<Vec<FusionOp>, usize>,
}

impl FusionEngine {
    pub fn new() -> Self {
        let mut engine = Self {
            patterns: Vec::new(),
            fusion_cache: HashMap::new(),
        };
        
        engine.register_default_patterns();
        engine
    }

    fn register_default_patterns(&mut self) {
        self.register_pattern(FusionPattern {
            ops: vec![FusionOp::MatMul, FusionOp::Add, FusionOp::ReLU],
            name: "gemm_bias_relu".to_string(),
            speedup_factor: 2.5,
        });

        self.register_pattern(FusionPattern {
            ops: vec![FusionOp::Conv2D, FusionOp::BatchNorm, FusionOp::ReLU],
            name: "conv_bn_relu".to_string(),
            speedup_factor: 3.0,
        });

        self.register_pattern(FusionPattern {
            ops: vec![FusionOp::MatMul, FusionOp::GELU],
            name: "gemm_gelu".to_string(),
            speedup_factor: 1.8,
        });

        self.register_pattern(FusionPattern {
            ops: vec![FusionOp::LayerNorm, FusionOp::MatMul],
            name: "ln_gemm".to_string(),
            speedup_factor: 1.6,
        });

        self.register_pattern(FusionPattern {
            ops: vec![FusionOp::MatMul, FusionOp::Add, FusionOp::LayerNorm],
            name: "gemm_add_ln".to_string(),
            speedup_factor: 2.2,
        });

        self.register_pattern(FusionPattern {
            ops: vec![FusionOp::MatMul, FusionOp::MatMul, FusionOp::Add],
            name: "multi_gemm_add".to_string(),
            speedup_factor: 2.8,
        });

        self.register_pattern(FusionPattern {
            ops: vec![FusionOp::Conv2D, FusionOp::Add, FusionOp::ReLU],
            name: "conv_residual_relu".to_string(),
            speedup_factor: 2.4,
        });

        self.register_pattern(FusionPattern {
            ops: vec![FusionOp::MatMul, FusionOp::Softmax],
            name: "attention_score".to_string(),
            speedup_factor: 1.9,
        });
    }

    pub fn register_pattern(&mut self, pattern: FusionPattern) {
        let idx = self.patterns.len();
        self.fusion_cache.insert(pattern.ops.clone(), idx);
        self.patterns.push(pattern);
    }

    pub fn find_fusion_opportunities(&self, op_sequence: &[FusionOp]) -> Vec<FusionMatch> {
        let mut matches = Vec::new();
        
        for pattern in &self.patterns {
            let mut i = 0;
            while i + pattern.ops.len() <= op_sequence.len() {
                if &op_sequence[i..i + pattern.ops.len()] == pattern.ops.as_slice() {
                    matches.push(FusionMatch {
                        pattern: pattern.clone(),
                        start_idx: i,
                        end_idx: i + pattern.ops.len(),
                    });
                    i += pattern.ops.len();
                } else {
                    i += 1;
                }
            }
        }
        
        matches
    }

    pub fn estimate_speedup(&self, op_sequence: &[FusionOp]) -> f32 {
        let matches = self.find_fusion_opportunities(op_sequence);
        
        if matches.is_empty() {
            return 1.0;
        }
        
        let total_speedup: f32 = matches.iter()
            .map(|m| m.pattern.speedup_factor)
            .product();
        
        total_speedup / matches.len() as f32
    }

    pub fn optimize_sequence(&self, op_sequence: &[FusionOp]) -> OptimizedSequence {
        let matches = self.find_fusion_opportunities(op_sequence);
        
        let mut optimized_ops = Vec::new();
        let mut covered = vec![false; op_sequence.len()];
        
        for fusion_match in &matches {
            for i in fusion_match.start_idx..fusion_match.end_idx {
                covered[i] = true;
            }
            optimized_ops.push(OptimizedOp::Fused(fusion_match.pattern.clone()));
        }
        
        for (i, &op) in op_sequence.iter().enumerate() {
            if !covered[i] {
                optimized_ops.push(OptimizedOp::Single(op));
            }
        }
        
        OptimizedSequence {
            ops: optimized_ops,
            original_count: op_sequence.len(),
            fused_count: matches.len(),
            estimated_speedup: self.estimate_speedup(op_sequence),
        }
    }

    pub fn can_fuse(&self, ops: &[FusionOp]) -> bool {
        self.fusion_cache.contains_key(ops)
    }

    pub fn get_pattern(&self, ops: &[FusionOp]) -> Option<&FusionPattern> {
        self.fusion_cache.get(ops).and_then(|&idx| self.patterns.get(idx))
    }
}

impl Default for FusionEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct FusionMatch {
    pub pattern: FusionPattern,
    pub start_idx: usize,
    pub end_idx: usize,
}

#[derive(Debug, Clone)]
pub enum OptimizedOp {
    Single(FusionOp),
    Fused(FusionPattern),
}

#[derive(Debug, Clone)]
pub struct OptimizedSequence {
    pub ops: Vec<OptimizedOp>,
    pub original_count: usize,
    pub fused_count: usize,
    pub estimated_speedup: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fusion_engine_creation() {
        let engine = FusionEngine::new();
        assert!(!engine.patterns.is_empty());
    }

    #[test]
    fn test_find_gemm_bias_relu() {
        let engine = FusionEngine::new();
        let ops = vec![FusionOp::MatMul, FusionOp::Add, FusionOp::ReLU];
        
        let matches = engine.find_fusion_opportunities(&ops);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].pattern.name, "gemm_bias_relu");
    }

    #[test]
    fn test_find_conv_bn_relu() {
        let engine = FusionEngine::new();
        let ops = vec![FusionOp::Conv2D, FusionOp::BatchNorm, FusionOp::ReLU];
        
        let matches = engine.find_fusion_opportunities(&ops);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].pattern.name, "conv_bn_relu");
    }

    #[test]
    fn test_multiple_fusions() {
        let engine = FusionEngine::new();
        let ops = vec![
            FusionOp::MatMul, FusionOp::Add, FusionOp::ReLU,
            FusionOp::Conv2D, FusionOp::BatchNorm, FusionOp::ReLU,
        ];
        
        let matches = engine.find_fusion_opportunities(&ops);
        assert_eq!(matches.len(), 2);
    }

    #[test]
    fn test_estimate_speedup() {
        let engine = FusionEngine::new();
        let ops = vec![FusionOp::MatMul, FusionOp::Add, FusionOp::ReLU];
        
        let speedup = engine.estimate_speedup(&ops);
        assert!(speedup > 1.0);
    }

    #[test]
    fn test_optimize_sequence() {
        let engine = FusionEngine::new();
        let ops = vec![
            FusionOp::MatMul, FusionOp::Add, FusionOp::ReLU,
            FusionOp::MaxPool,
        ];
        
        let optimized = engine.optimize_sequence(&ops);
        assert_eq!(optimized.original_count, 4);
        assert_eq!(optimized.fused_count, 1);
        assert!(optimized.estimated_speedup > 1.0);
    }

    #[test]
    fn test_can_fuse() {
        let engine = FusionEngine::new();
        let ops = vec![FusionOp::MatMul, FusionOp::Add, FusionOp::ReLU];
        
        assert!(engine.can_fuse(&ops));
        
        let non_fusible = vec![FusionOp::MaxPool, FusionOp::Dropout];
        assert!(!engine.can_fuse(&non_fusible));
    }

    #[test]
    fn test_get_pattern() {
        let engine = FusionEngine::new();
        let ops = vec![FusionOp::MatMul, FusionOp::Add, FusionOp::ReLU];
        
        let pattern = engine.get_pattern(&ops);
        assert!(pattern.is_some());
        assert_eq!(pattern.unwrap().name, "gemm_bias_relu");
    }

    #[test]
    fn test_attention_fusion() {
        let engine = FusionEngine::new();
        let ops = vec![FusionOp::MatMul, FusionOp::Softmax];
        
        let matches = engine.find_fusion_opportunities(&ops);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].pattern.name, "attention_score");
    }
}
