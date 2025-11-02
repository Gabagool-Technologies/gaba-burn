use std::time::Duration;

#[derive(Debug, Clone)]
pub struct LSTMState {
    pub hidden: Vec<f32>,
    pub cell: Vec<f32>,
}

impl LSTMState {
    pub fn new(hidden_size: usize) -> Self {
        Self {
            hidden: vec![0.0; hidden_size],
            cell: vec![0.0; hidden_size],
        }
    }
    
    pub fn zero(&mut self) {
        for h in &mut self.hidden {
            *h = 0.0;
        }
        for c in &mut self.cell {
            *c = 0.0;
        }
    }
}

pub struct LSTMParams {
    pub input_size: usize,
    pub hidden_size: usize,
    
    pub w_ii: Vec<f32>,
    pub w_if: Vec<f32>,
    pub w_ig: Vec<f32>,
    pub w_io: Vec<f32>,
    
    pub w_hi: Vec<f32>,
    pub w_hf: Vec<f32>,
    pub w_hg: Vec<f32>,
    pub w_ho: Vec<f32>,
    
    pub b_ii: Vec<f32>,
    pub b_if: Vec<f32>,
    pub b_ig: Vec<f32>,
    pub b_io: Vec<f32>,
    
    pub b_hi: Vec<f32>,
    pub b_hf: Vec<f32>,
    pub b_hg: Vec<f32>,
    pub b_ho: Vec<f32>,
}

impl LSTMParams {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let init_weight = |size: usize| -> Vec<f32> {
            let scale = (2.0 / size as f32).sqrt();
            (0..size).map(|i| {
                let x = (i as f32 * 0.1) % 1.0;
                (x - 0.5) * scale
            }).collect()
        };
        
        let init_bias = |size: usize| -> Vec<f32> {
            vec![0.0; size]
        };
        
        Self {
            input_size,
            hidden_size,
            
            w_ii: init_weight(input_size * hidden_size),
            w_if: init_weight(input_size * hidden_size),
            w_ig: init_weight(input_size * hidden_size),
            w_io: init_weight(input_size * hidden_size),
            
            w_hi: init_weight(hidden_size * hidden_size),
            w_hf: init_weight(hidden_size * hidden_size),
            w_hg: init_weight(hidden_size * hidden_size),
            w_ho: init_weight(hidden_size * hidden_size),
            
            b_ii: init_bias(hidden_size),
            b_if: init_bias(hidden_size),
            b_ig: init_bias(hidden_size),
            b_io: init_bias(hidden_size),
            
            b_hi: init_bias(hidden_size),
            b_hf: init_bias(hidden_size),
            b_hg: init_bias(hidden_size),
            b_ho: init_bias(hidden_size),
        }
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn tanh(x: f32) -> f32 {
    x.tanh()
}

fn matmul_vec(weights: &[f32], input: &[f32], output: &mut [f32], rows: usize, cols: usize) {
    for i in 0..rows {
        let mut sum = 0.0;
        for j in 0..cols {
            sum += weights[i * cols + j] * input[j];
        }
        output[i] = sum;
    }
}

pub fn lstm_cell_forward(
    input: &[f32],
    state: &LSTMState,
    params: &LSTMParams,
    new_state: &mut LSTMState,
) -> Duration {
    let start = std::time::Instant::now();
    
    let hidden_size = params.hidden_size;
    let input_size = params.input_size;
    
    let mut i_gate = vec![0.0; hidden_size];
    let mut f_gate = vec![0.0; hidden_size];
    let mut g_gate = vec![0.0; hidden_size];
    let mut o_gate = vec![0.0; hidden_size];
    
    let mut tmp = vec![0.0; hidden_size];
    
    matmul_vec(&params.w_ii, input, &mut i_gate, hidden_size, input_size);
    matmul_vec(&params.w_hi, &state.hidden, &mut tmp, hidden_size, hidden_size);
    for i in 0..hidden_size {
        i_gate[i] = sigmoid(i_gate[i] + tmp[i] + params.b_ii[i] + params.b_hi[i]);
    }
    
    matmul_vec(&params.w_if, input, &mut f_gate, hidden_size, input_size);
    matmul_vec(&params.w_hf, &state.hidden, &mut tmp, hidden_size, hidden_size);
    for i in 0..hidden_size {
        f_gate[i] = sigmoid(f_gate[i] + tmp[i] + params.b_if[i] + params.b_hf[i]);
    }
    
    matmul_vec(&params.w_ig, input, &mut g_gate, hidden_size, input_size);
    matmul_vec(&params.w_hg, &state.hidden, &mut tmp, hidden_size, hidden_size);
    for i in 0..hidden_size {
        g_gate[i] = tanh(g_gate[i] + tmp[i] + params.b_ig[i] + params.b_hg[i]);
    }
    
    matmul_vec(&params.w_io, input, &mut o_gate, hidden_size, input_size);
    matmul_vec(&params.w_ho, &state.hidden, &mut tmp, hidden_size, hidden_size);
    for i in 0..hidden_size {
        o_gate[i] = sigmoid(o_gate[i] + tmp[i] + params.b_io[i] + params.b_ho[i]);
    }
    
    for i in 0..hidden_size {
        new_state.cell[i] = f_gate[i] * state.cell[i] + i_gate[i] * g_gate[i];
        new_state.hidden[i] = o_gate[i] * tanh(new_state.cell[i]);
    }
    
    start.elapsed()
}

pub fn lstm_forward(
    input_sequence: &[Vec<f32>],
    initial_state: &LSTMState,
    params: &LSTMParams,
    output_sequence: &mut [Vec<f32>],
) -> Duration {
    let start = std::time::Instant::now();
    
    let mut state = initial_state.clone();
    
    for (t, input) in input_sequence.iter().enumerate() {
        let mut new_state = LSTMState::new(params.hidden_size);
        lstm_cell_forward(input, &state, params, &mut new_state);
        
        output_sequence[t] = new_state.hidden.clone();
        state = new_state;
    }
    
    start.elapsed()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_lstm_cell() {
        let input_size = 10;
        let hidden_size = 20;
        
        let params = LSTMParams::new(input_size, hidden_size);
        let state = LSTMState::new(hidden_size);
        let mut new_state = LSTMState::new(hidden_size);
        
        let input = vec![1.0; input_size];
        
        let duration = lstm_cell_forward(&input, &state, &params, &mut new_state);
        
        assert!(duration.as_micros() > 0);
        assert_eq!(new_state.hidden.len(), hidden_size);
        assert_eq!(new_state.cell.len(), hidden_size);
    }
    
    #[test]
    fn test_lstm_forward() {
        let input_size = 10;
        let hidden_size = 20;
        let seq_len = 5;
        
        let params = LSTMParams::new(input_size, hidden_size);
        let state = LSTMState::new(hidden_size);
        
        let input_sequence: Vec<Vec<f32>> = (0..seq_len)
            .map(|_| vec![1.0; input_size])
            .collect();
        
        let mut output_sequence: Vec<Vec<f32>> = (0..seq_len)
            .map(|_| vec![0.0; hidden_size])
            .collect();
        
        let duration = lstm_forward(&input_sequence, &state, &params, &mut output_sequence);
        
        assert!(duration.as_micros() > 0);
        assert_eq!(output_sequence.len(), seq_len);
    }
}
