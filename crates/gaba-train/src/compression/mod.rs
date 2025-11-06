pub mod low_rank;
pub mod distillation;

pub use low_rank::{LowRankDecomposition, DecompositionMethod};
pub use distillation::{KnowledgeDistillation, DistillationLoss};
