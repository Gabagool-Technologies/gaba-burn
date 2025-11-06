/// Transformer Architecture Components
/// Multi-head attention and transformer blocks for sequence modeling

pub struct MultiHeadAttentionParams {
    pub num_heads: usize,
    pub head_dim: usize,
    pub q_weights: Vec<f32>,
    pub k_weights: Vec<f32>,
    pub v_weights: Vec<f32>,
    pub out_weights: Vec<f32>,
}

impl MultiHeadAttentionParams {
    pub fn new(embed_dim: usize, num_heads: usize) -> Self {
        assert_eq!(
            embed_dim % num_heads,
            0,
            "embed_dim must be divisible by num_heads"
        );
        let head_dim = embed_dim / num_heads;
        let total_params = embed_dim * embed_dim;

        Self {
            num_heads,
            head_dim,
            q_weights: vec![0.01; total_params],
            k_weights: vec![0.01; total_params],
            v_weights: vec![0.01; total_params],
            out_weights: vec![0.01; total_params],
        }
    }
}

pub fn multi_head_attention_forward(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    params: &MultiHeadAttentionParams,
    seq_len: usize,
    embed_dim: usize,
) -> Vec<f32> {
    let num_heads = params.num_heads;
    let head_dim = params.head_dim;

    // Project Q, K, V
    let q_proj = matmul(query, &params.q_weights, seq_len, embed_dim, embed_dim);
    let k_proj = matmul(key, &params.k_weights, seq_len, embed_dim, embed_dim);
    let v_proj = matmul(value, &params.v_weights, seq_len, embed_dim, embed_dim);

    // Reshape for multi-head: [seq_len, embed_dim] -> [num_heads, seq_len, head_dim]
    let mut output = vec![0.0; seq_len * embed_dim];

    for h in 0..num_heads {
        // Extract head
        let mut q_head = vec![0.0; seq_len * head_dim];
        let mut k_head = vec![0.0; seq_len * head_dim];
        let mut v_head = vec![0.0; seq_len * head_dim];

        for i in 0..seq_len {
            for j in 0..head_dim {
                let src_idx = i * embed_dim + h * head_dim + j;
                let dst_idx = i * head_dim + j;
                q_head[dst_idx] = q_proj[src_idx];
                k_head[dst_idx] = k_proj[src_idx];
                v_head[dst_idx] = v_proj[src_idx];
            }
        }

        // Compute attention: Q @ K^T / sqrt(head_dim)
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut scores = vec![0.0; seq_len * seq_len];

        for i in 0..seq_len {
            for j in 0..seq_len {
                let mut score = 0.0;
                for k in 0..head_dim {
                    score += q_head[i * head_dim + k] * k_head[j * head_dim + k];
                }
                scores[i * seq_len + j] = score * scale;
            }
        }

        // Softmax
        for i in 0..seq_len {
            let row_start = i * seq_len;
            let row_end = row_start + seq_len;
            let row = &mut scores[row_start..row_end];

            let max_val = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut sum = 0.0;
            for val in row.iter_mut() {
                *val = (*val - max_val).exp();
                sum += *val;
            }
            for val in row.iter_mut() {
                *val /= sum;
            }
        }

        // Attention @ V
        let mut head_output = vec![0.0; seq_len * head_dim];
        for i in 0..seq_len {
            for j in 0..head_dim {
                let mut sum = 0.0;
                for k in 0..seq_len {
                    sum += scores[i * seq_len + k] * v_head[k * head_dim + j];
                }
                head_output[i * head_dim + j] = sum;
            }
        }

        // Concatenate heads
        for i in 0..seq_len {
            for j in 0..head_dim {
                output[i * embed_dim + h * head_dim + j] = head_output[i * head_dim + j];
            }
        }
    }

    // Output projection
    matmul(&output, &params.out_weights, seq_len, embed_dim, embed_dim)
}

pub struct TransformerBlockParams {
    pub attention: MultiHeadAttentionParams,
    pub ff_w1: Vec<f32>,
    pub ff_b1: Vec<f32>,
    pub ff_w2: Vec<f32>,
    pub ff_b2: Vec<f32>,
    pub ln1_gamma: Vec<f32>,
    pub ln1_beta: Vec<f32>,
    pub ln2_gamma: Vec<f32>,
    pub ln2_beta: Vec<f32>,
}

impl TransformerBlockParams {
    pub fn new(embed_dim: usize, num_heads: usize, ff_dim: usize) -> Self {
        Self {
            attention: MultiHeadAttentionParams::new(embed_dim, num_heads),
            ff_w1: vec![0.01; embed_dim * ff_dim],
            ff_b1: vec![0.0; ff_dim],
            ff_w2: vec![0.01; ff_dim * embed_dim],
            ff_b2: vec![0.0; embed_dim],
            ln1_gamma: vec![1.0; embed_dim],
            ln1_beta: vec![0.0; embed_dim],
            ln2_gamma: vec![1.0; embed_dim],
            ln2_beta: vec![0.0; embed_dim],
        }
    }
}

pub fn transformer_block_forward(
    input: &[f32],
    params: &TransformerBlockParams,
    seq_len: usize,
    embed_dim: usize,
) -> Vec<f32> {
    // Self-attention with residual
    let attn_out =
        multi_head_attention_forward(input, input, input, &params.attention, seq_len, embed_dim);

    let mut residual1 = vec![0.0; input.len()];
    for i in 0..input.len() {
        residual1[i] = input[i] + attn_out[i];
    }

    // Layer norm 1
    let ln1_out = layer_norm(
        &residual1,
        seq_len,
        embed_dim,
        &params.ln1_gamma,
        &params.ln1_beta,
    );

    // Feed-forward with residual
    let ff_dim = params.ff_b1.len();
    let ff1 = matmul(&ln1_out, &params.ff_w1, seq_len, embed_dim, ff_dim);
    let mut ff1_relu = ff1.clone();
    for val in ff1_relu.iter_mut() {
        *val = val.max(0.0); // ReLU
    }

    let ff2 = matmul(&ff1_relu, &params.ff_w2, seq_len, ff_dim, embed_dim);

    let mut residual2 = vec![0.0; ln1_out.len()];
    for i in 0..ln1_out.len() {
        residual2[i] = ln1_out[i] + ff2[i];
    }

    // Layer norm 2
    layer_norm(
        &residual2,
        seq_len,
        embed_dim,
        &params.ln2_gamma,
        &params.ln2_beta,
    )
}

fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

fn layer_norm(
    input: &[f32],
    seq_len: usize,
    embed_dim: usize,
    gamma: &[f32],
    beta: &[f32],
) -> Vec<f32> {
    let mut output = vec![0.0; input.len()];
    let epsilon = 1e-5;

    for i in 0..seq_len {
        let start = i * embed_dim;
        let end = start + embed_dim;
        let slice = &input[start..end];

        let mean = slice.iter().sum::<f32>() / embed_dim as f32;
        let variance = slice.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / embed_dim as f32;
        let std = (variance + epsilon).sqrt();

        for j in 0..embed_dim {
            let normalized = (slice[j] - mean) / std;
            output[start + j] = gamma[j] * normalized + beta[j];
        }
    }

    output
}

pub fn positional_encoding(seq_len: usize, embed_dim: usize) -> Vec<f32> {
    let mut encoding = vec![0.0; seq_len * embed_dim];

    for pos in 0..seq_len {
        for i in 0..embed_dim {
            let angle = pos as f32 / 10000_f32.powf(2.0 * (i / 2) as f32 / embed_dim as f32);
            encoding[pos * embed_dim + i] = if i % 2 == 0 { angle.sin() } else { angle.cos() };
        }
    }

    encoding
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_head_attention() {
        let seq_len = 4;
        let embed_dim = 8;
        let num_heads = 2;

        let input = vec![0.1; seq_len * embed_dim];
        let params = MultiHeadAttentionParams::new(embed_dim, num_heads);

        let output =
            multi_head_attention_forward(&input, &input, &input, &params, seq_len, embed_dim);

        assert_eq!(output.len(), seq_len * embed_dim);
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_transformer_block() {
        let seq_len = 4;
        let embed_dim = 8;
        let num_heads = 2;
        let ff_dim = 16;

        let input = vec![0.1; seq_len * embed_dim];
        let params = TransformerBlockParams::new(embed_dim, num_heads, ff_dim);

        let output = transformer_block_forward(&input, &params, seq_len, embed_dim);

        assert_eq!(output.len(), seq_len * embed_dim);
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_positional_encoding() {
        let seq_len = 10;
        let embed_dim = 8;

        let encoding = positional_encoding(seq_len, embed_dim);

        assert_eq!(encoding.len(), seq_len * embed_dim);
        assert!(encoding.iter().all(|&x| x.is_finite()));
    }
}
