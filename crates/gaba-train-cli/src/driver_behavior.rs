//! Driver Behavior Analysis

#[allow(dead_code)]
pub struct DriverBehaviorAnalyzer {
    weights: Vec<f32>,
}

impl DriverBehaviorAnalyzer {
    pub fn new() -> Self {
        Self {
            weights: vec![-0.5, -0.4, -0.3, 0.2, 0.3, -0.2, 0.4, -0.1],
        }
    }

    pub fn analyze(&self, behavior: &DriverBehavior) -> f32 {
        let features = behavior.to_features();
        features.iter().zip(self.weights.iter()).map(|(f, w)| f * w).sum::<f32>()
    }
}

#[allow(dead_code)]
pub struct DriverBehavior {
    pub harsh_braking: f32,
    pub rapid_accel: f32,
    pub speeding: u32,
    pub smooth_score: f32,
    pub fuel_eff: f32,
    pub idle_ratio: f32,
    pub on_time_rate: f32,
    pub accidents: u32,
}

impl DriverBehavior {
    pub fn to_features(&self) -> Vec<f32> {
        vec![
            self.harsh_braking, self.rapid_accel, self.speeding as f32,
            self.smooth_score, self.fuel_eff, self.idle_ratio,
            self.on_time_rate, self.accidents as f32,
        ]
    }
}
