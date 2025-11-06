use ndarray::Array1;

#[derive(Clone, Debug)]
pub enum DistillationLoss {
    KLDivergence,
    ReverseKLD,
    MSE,
    CosineEmbedding,
}

pub struct KnowledgeDistillation {
    temperature: f32,
    alpha: f32,
    loss_type: DistillationLoss,
}

impl KnowledgeDistillation {
    pub fn new(temperature: f32, alpha: f32) -> Self {
        Self {
            temperature,
            alpha,
            loss_type: DistillationLoss::KLDivergence,
        }
    }
    
    pub fn with_loss_type(mut self, loss_type: DistillationLoss) -> Self {
        self.loss_type = loss_type;
        self
    }
    
    pub fn compute_loss(
        &self,
        student_logits: &Array1<f32>,
        teacher_logits: &Array1<f32>,
        hard_targets: Option<&Array1<f32>>,
    ) -> f32 {
        let soft_loss = match self.loss_type {
            DistillationLoss::KLDivergence => {
                self.kl_divergence_loss(student_logits, teacher_logits)
            }
            DistillationLoss::ReverseKLD => {
                self.reverse_kl_loss(student_logits, teacher_logits)
            }
            DistillationLoss::MSE => {
                self.mse_loss(student_logits, teacher_logits)
            }
            DistillationLoss::CosineEmbedding => {
                self.cosine_loss(student_logits, teacher_logits)
            }
        };
        
        if let Some(targets) = hard_targets {
            let hard_loss = self.cross_entropy_loss(student_logits, targets);
            self.alpha * hard_loss + (1.0 - self.alpha) * soft_loss
        } else {
            soft_loss
        }
    }
    
    fn softmax_with_temperature(&self, logits: &Array1<f32>) -> Array1<f32> {
        let scaled = logits.mapv(|x| x / self.temperature);
        let max_val = scaled.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_vals = scaled.mapv(|x| (x - max_val).exp());
        let sum_exp = exp_vals.sum();
        exp_vals.mapv(|x| x / sum_exp)
    }
    
    fn kl_divergence_loss(&self, student_logits: &Array1<f32>, teacher_logits: &Array1<f32>) -> f32 {
        let student_probs = self.softmax_with_temperature(student_logits);
        let teacher_probs = self.softmax_with_temperature(teacher_logits);
        
        let kl: f32 = teacher_probs.iter()
            .zip(student_probs.iter())
            .map(|(t, s)| {
                if *t > 1e-10 {
                    t * (t / s.max(1e-10)).ln()
                } else {
                    0.0
                }
            })
            .sum();
        
        kl * self.temperature * self.temperature
    }
    
    fn reverse_kl_loss(&self, student_logits: &Array1<f32>, teacher_logits: &Array1<f32>) -> f32 {
        let student_probs = self.softmax_with_temperature(student_logits);
        let teacher_probs = self.softmax_with_temperature(teacher_logits);
        
        let rkl: f32 = student_probs.iter()
            .zip(teacher_probs.iter())
            .map(|(s, t)| {
                if *s > 1e-10 {
                    s * (s / t.max(1e-10)).ln()
                } else {
                    0.0
                }
            })
            .sum();
        
        rkl * self.temperature * self.temperature
    }
    
    fn mse_loss(&self, student_logits: &Array1<f32>, teacher_logits: &Array1<f32>) -> f32 {
        let diff = student_logits - teacher_logits;
        (&diff * &diff).sum() / student_logits.len() as f32
    }
    
    fn cosine_loss(&self, student_logits: &Array1<f32>, teacher_logits: &Array1<f32>) -> f32 {
        let dot_product: f32 = student_logits.iter()
            .zip(teacher_logits.iter())
            .map(|(s, t)| s * t)
            .sum();
        
        let student_norm = student_logits.mapv(|x| x * x).sum().sqrt();
        let teacher_norm = teacher_logits.mapv(|x| x * x).sum().sqrt();
        
        1.0 - (dot_product / (student_norm * teacher_norm + 1e-10))
    }
    
    fn cross_entropy_loss(&self, logits: &Array1<f32>, targets: &Array1<f32>) -> f32 {
        let probs = self.softmax_with_temperature(logits);
        let ce: f32 = targets.iter()
            .zip(probs.iter())
            .map(|(t, p)| -t * p.max(1e-10).ln())
            .sum();
        ce
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_kd_basic() {
        let kd = KnowledgeDistillation::new(2.0, 0.5);
        
        let student = arr1(&[1.0, 2.0, 3.0]);
        let teacher = arr1(&[1.5, 2.5, 3.5]);
        
        let loss = kd.compute_loss(&student, &teacher, None);
        assert!(loss >= 0.0);
    }
    
    #[test]
    fn test_kd_with_hard_targets() {
        let kd = KnowledgeDistillation::new(2.0, 0.7);
        
        let student = arr1(&[1.0, 2.0, 3.0]);
        let teacher = arr1(&[1.5, 2.5, 3.5]);
        let targets = arr1(&[0.0, 0.0, 1.0]);
        
        let loss = kd.compute_loss(&student, &teacher, Some(&targets));
        assert!(loss >= 0.0);
    }
    
    #[test]
    fn test_reverse_kld() {
        let kd = KnowledgeDistillation::new(1.0, 0.5)
            .with_loss_type(DistillationLoss::ReverseKLD);
        
        let student = arr1(&[1.0, 2.0, 3.0]);
        let teacher = arr1(&[1.0, 2.0, 3.0]);
        
        let loss = kd.compute_loss(&student, &teacher, None);
        assert_relative_eq!(loss, 0.0, epsilon = 1e-4);
    }
    
    #[test]
    fn test_mse_loss() {
        let kd = KnowledgeDistillation::new(1.0, 0.5)
            .with_loss_type(DistillationLoss::MSE);
        
        let student = arr1(&[1.0, 2.0, 3.0]);
        let teacher = arr1(&[1.0, 2.0, 3.0]);
        
        let loss = kd.compute_loss(&student, &teacher, None);
        assert_relative_eq!(loss, 0.0, epsilon = 1e-6);
    }
}
