//! Rotary Position Embedding (RoPE) implementation
//! 
//! This module implements Rotary Position Embedding, which encodes positional
//! information by rotating query and key vectors in a way that naturally
//! incorporates relative position information.

use candle_core::{Tensor, Device, DType, Result as CandleResult};
use std::f64::consts::PI;

/// Apply rotary embedding to query and key tensors
pub fn apply_rotary_emb(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> CandleResult<(Tensor, Tensor)> {
    let rotated_q = apply_rotary_emb_single(q, cos, sin)?;
    let rotated_k = apply_rotary_emb_single(k, cos, sin)?;
    Ok((rotated_q, rotated_k))
}

/// Apply rotary embedding to a single tensor
pub fn apply_rotary_emb_single(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> CandleResult<Tensor> {
    // Split the last dimension in half
    let last_dim = x.dims().len() - 1;
    let dim_size = x.dim(last_dim)?;
    let half_dim = dim_size / 2;
    
    // Split into two halves: x1 and x2
    let x1 = x.narrow(last_dim, 0, half_dim)?;
    let x2 = x.narrow(last_dim, half_dim, half_dim)?;
    
    // Apply rotation: x1 * cos - x2 * sin, x2 * cos + x1 * sin
    let cos_x1 = x1.broadcast_mul(cos)?;
    let sin_x2 = x2.broadcast_mul(sin)?;
    let cos_x2 = x2.broadcast_mul(cos)?;
    let sin_x1 = x1.broadcast_mul(sin)?;
    
    let rotated_x1 = (cos_x1 - sin_x2)?;
    let rotated_x2 = (cos_x2 + sin_x1)?;
    
    // Concatenate the rotated halves
    Tensor::cat(&[rotated_x1, rotated_x2], last_dim)
}

/// Rotary Position Embedding layer
#[derive(Debug)]
pub struct RotaryEmbedding {
    /// Dimension of each head
    head_dim: usize,
    
    /// Maximum sequence length supported
    max_position_embeddings: usize,
    
    /// Base frequency for the rotation
    base: f64,
    
    /// Precomputed cosine values
    cos_cache: Tensor,
    
    /// Precomputed sine values
    sin_cache: Tensor,
    
    /// Device for computations
    device: Device,
}

impl RotaryEmbedding {
    /// Create a new rotary embedding layer
    pub fn new(
        head_dim: usize,
        max_position_embeddings: usize,
        base: f64,
        device: &Device,
        dtype: DType,
    ) -> CandleResult<Self> {
        // Compute inverse frequencies
        let dim_range: Vec<f64> = (0..head_dim)
            .step_by(2)
            .map(|i| i as f64)
            .collect();
        
        let inv_freq: Vec<f64> = dim_range
            .iter()
            .map(|&i| 1.0 / base.powf(i / head_dim as f64))
            .collect();
        
        let inv_freq_tensor = Tensor::from_vec(inv_freq, (head_dim / 2,), device)?
            .to_dtype(dtype)?;
        
        // Create position indices
        let positions: Vec<f64> = (0..max_position_embeddings)
            .map(|i| i as f64)
            .collect();
        let positions_tensor = Tensor::from_vec(positions, (max_position_embeddings,), device)?
            .to_dtype(dtype)?;
        
        // Compute frequencies: positions ⊗ inv_freq
        let freqs = positions_tensor
            .unsqueeze(1)?
            .broadcast_mul(&inv_freq_tensor.unsqueeze(0)?)?;
        
        // Compute cos and sin
        let cos_cache = freqs.cos()?;
        let sin_cache = freqs.sin()?;
        
        Ok(Self {
            head_dim,
            max_position_embeddings,
            base,
            cos_cache,
            sin_cache,
            device: device.clone(),
        })
    }
    
    /// Create rotary embedding with scaling
    pub fn new_with_scaling(
        head_dim: usize,
        max_position_embeddings: usize,
        base: f64,
        scaling_factor: f64,
        device: &Device,
        dtype: DType,
    ) -> CandleResult<Self> {
        // Apply scaling to the base frequency
        let scaled_base = base * scaling_factor;
        Self::new(head_dim, max_position_embeddings, scaled_base, device, dtype)
    }
    
    /// Get cosine and sine values for given positions
    pub fn get_cos_sin(&self, positions: &Tensor) -> CandleResult<(Tensor, Tensor)> {
        // Index into the precomputed cache
        let cos = self.cos_cache.index_select(positions, 0)?;
        let sin = self.sin_cache.index_select(positions, 0)?;
        
        Ok((cos, sin))
    }
    
    /// Forward pass: apply rotary embedding to query and key
    pub fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        positions: &Tensor,
    ) -> CandleResult<(Tensor, Tensor)> {
        let (cos, sin) = self.get_cos_sin(positions)?;
        
        // Expand dimensions to match q and k
        let cos = self.expand_for_attention(&cos, q)?;
        let sin = self.expand_for_attention(&sin, q)?;
        
        apply_rotary_emb(q, k, &cos, &sin)
    }
    
    /// Expand cos/sin tensors to match attention tensor dimensions
    fn expand_for_attention(&self, tensor: &Tensor, reference: &Tensor) -> CandleResult<Tensor> {
        let ref_dims = reference.dims();
        let tensor_dims = tensor.dims();
        
        // tensor is typically [seq_len, head_dim/2]
        // reference is typically [batch_size, seq_len, num_heads, head_dim]
        // We need to expand to [batch_size, seq_len, num_heads, head_dim/2]
        
        match ref_dims.len() {
            3 => {
                // [seq_len, num_heads, head_dim] format
                let expanded = tensor
                    .unsqueeze(1)?  // [seq_len, 1, head_dim/2]
                    .broadcast_as((ref_dims[0], ref_dims[1], tensor_dims[1]))?;
                Ok(expanded)
            }
            4 => {
                // [batch_size, seq_len, num_heads, head_dim] format
                let expanded = tensor
                    .unsqueeze(0)?  // [1, seq_len, head_dim/2]
                    .unsqueeze(2)?  // [1, seq_len, 1, head_dim/2]
                    .broadcast_as((ref_dims[0], ref_dims[1], ref_dims[2], tensor_dims[1]))?;
                Ok(expanded)
            }
            _ => {
                anyhow::bail!("Unsupported tensor dimensions for rotary embedding: {:?}", ref_dims);
            }
        }
    }
    
    /// Get the head dimension
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }
    
    /// Get the maximum position embeddings
    pub fn max_position_embeddings(&self) -> usize {
        self.max_position_embeddings
    }
    
    /// Get the base frequency
    pub fn base(&self) -> f64 {
        self.base
    }
}

/// Optimized rotary embedding with better memory usage
#[derive(Debug)]
pub struct OptimizedRotaryEmbedding {
    head_dim: usize,
    max_position_embeddings: usize,
    base: f64,
    inv_freq: Tensor,
    device: Device,
}

impl OptimizedRotaryEmbedding {
    /// Create a new optimized rotary embedding
    pub fn new(
        head_dim: usize,
        max_position_embeddings: usize,
        base: f64,
        device: &Device,
        dtype: DType,
    ) -> CandleResult<Self> {
        // Only store inverse frequencies, compute cos/sin on demand
        let dim_range: Vec<f64> = (0..head_dim)
            .step_by(2)
            .map(|i| i as f64)
            .collect();
        
        let inv_freq: Vec<f64> = dim_range
            .iter()
            .map(|&i| 1.0 / base.powf(i / head_dim as f64))
            .collect();
        
        let inv_freq_tensor = Tensor::from_vec(inv_freq, (head_dim / 2,), device)?
            .to_dtype(dtype)?;
        
        Ok(Self {
            head_dim,
            max_position_embeddings,
            base,
            inv_freq: inv_freq_tensor,
            device: device.clone(),
        })
    }
    
    /// Compute cos and sin on demand for given positions
    pub fn compute_cos_sin(&self, positions: &Tensor) -> CandleResult<(Tensor, Tensor)> {
        // Convert positions to the same dtype as inv_freq
        let positions_f = positions.to_dtype(self.inv_freq.dtype())?;
        
        // Compute frequencies: positions ⊗ inv_freq
        let freqs = positions_f
            .unsqueeze(1)?
            .broadcast_mul(&self.inv_freq.unsqueeze(0)?)?;
        
        let cos = freqs.cos()?;
        let sin = freqs.sin()?;
        
        Ok((cos, sin))
    }
    
    /// Forward pass with on-demand computation
    pub fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        positions: &Tensor,
    ) -> CandleResult<(Tensor, Tensor)> {
        let (cos, sin) = self.compute_cos_sin(positions)?;
        
        // Expand to match q and k dimensions
        let cos_expanded = self.expand_for_attention(&cos, q)?;
        let sin_expanded = self.expand_for_attention(&sin, q)?;
        
        apply_rotary_emb(q, k, &cos_expanded, &sin_expanded)
    }
    
    /// Expand cos/sin tensors to match attention tensor dimensions
    fn expand_for_attention(&self, tensor: &Tensor, reference: &Tensor) -> CandleResult<Tensor> {
        let ref_dims = reference.dims();
        let tensor_dims = tensor.dims();
        
        match ref_dims.len() {
            3 => {
                tensor
                    .unsqueeze(1)?
                    .broadcast_as((ref_dims[0], ref_dims[1], tensor_dims[1]))
            }
            4 => {
                tensor
                    .unsqueeze(0)?
                    .unsqueeze(2)?
                    .broadcast_as((ref_dims[0], ref_dims[1], ref_dims[2], tensor_dims[1]))
            }
            _ => {
                anyhow::bail!("Unsupported tensor dimensions: {:?}", ref_dims);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, DType};
    
    #[test]
    fn test_rotary_embedding_creation() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        
        let rope = RotaryEmbedding::new(64, 2048, 10000.0, &device, dtype).unwrap();
        assert_eq!(rope.head_dim(), 64);
        assert_eq!(rope.max_position_embeddings(), 2048);
        assert_eq!(rope.base(), 10000.0);
    }
    
    #[test]
    fn test_apply_rotary_emb_single() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        
        // Create test tensors
        let x = Tensor::randn(0f32, 1f32, (2, 4), &device).unwrap(); // [seq_len, head_dim]
        let cos = Tensor::ones((2, 2), dtype, &device).unwrap(); // [seq_len, head_dim/2]
        let sin = Tensor::zeros((2, 2), dtype, &device).unwrap();
        
        let result = apply_rotary_emb_single(&x, &cos, &sin).unwrap();
        assert_eq!(result.dims(), x.dims());
        
        // With sin=0 and cos=1, the result should be the same as input
        let diff = (&result - &x).unwrap().abs().unwrap().sum_all().unwrap();
        let diff_val: f32 = diff.to_scalar().unwrap();
        assert!(diff_val < 1e-6, "With cos=1, sin=0, result should equal input");
    }
    
    #[test]
    fn test_rotary_embedding_forward() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        
        let rope = RotaryEmbedding::new(4, 10, 10000.0, &device, dtype).unwrap();
        
        // Create query and key tensors
        let q = Tensor::randn(0f32, 1f32, (2, 4), &device).unwrap(); // [seq_len, head_dim]
        let k = Tensor::randn(0f32, 1f32, (2, 4), &device).unwrap();
        let positions = Tensor::from_vec(vec![0u32, 1u32], (2,), &device).unwrap();
        
        let (rotated_q, rotated_k) = rope.forward(&q, &k, &positions).unwrap();
        
        assert_eq!(rotated_q.dims(), q.dims());
        assert_eq!(rotated_k.dims(), k.dims());
        
        // Results should be different from original (unless at position 0 with specific values)
        let q_diff = (&rotated_q - &q).unwrap().abs().unwrap().sum_all().unwrap();
        let k_diff = (&rotated_k - &k).unwrap().abs().unwrap().sum_all().unwrap();
        
        // At least one should be different (position 1 should definitely be rotated)
        let q_diff_val: f32 = q_diff.to_scalar().unwrap();
        let k_diff_val: f32 = k_diff.to_scalar().unwrap();
        assert!(q_diff_val > 1e-6 || k_diff_val > 1e-6, "Rotary embedding should change the tensors");
    }
    
    #[test]
    fn test_get_cos_sin() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        
        let rope = RotaryEmbedding::new(4, 10, 10000.0, &device, dtype).unwrap();
        let positions = Tensor::from_vec(vec![0u32, 1u32, 2u32], (3,), &device).unwrap();
        
        let (cos, sin) = rope.get_cos_sin(&positions).unwrap();
        
        assert_eq!(cos.dims(), [3, 2]); // [seq_len, head_dim/2]
        assert_eq!(sin.dims(), [3, 2]);
        
        // cos(0) should be 1, sin(0) should be 0
        let cos_0: f32 = cos.get(0).unwrap().get(0).unwrap().to_scalar().unwrap();
        let sin_0: f32 = sin.get(0).unwrap().get(0).unwrap().to_scalar().unwrap();
        
        assert!((cos_0 - 1.0).abs() < 1e-6, "cos(0) should be 1");
        assert!(sin_0.abs() < 1e-6, "sin(0) should be 0");
    }
    
    #[test]
    fn test_optimized_rotary_embedding() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        
        let rope = OptimizedRotaryEmbedding::new(4, 10, 10000.0, &device, dtype).unwrap();
        
        let q = Tensor::randn(0f32, 1f32, (2, 4), &device).unwrap();
        let k = Tensor::randn(0f32, 1f32, (2, 4), &device).unwrap();
        let positions = Tensor::from_vec(vec![0u32, 1u32], (2,), &device).unwrap();
        
        let (rotated_q, rotated_k) = rope.forward(&q, &k, &positions).unwrap();
        
        assert_eq!(rotated_q.dims(), q.dims());
        assert_eq!(rotated_k.dims(), k.dims());
    }
    
    #[test]
    fn test_rotary_embedding_with_scaling() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        
        let rope = RotaryEmbedding::new_with_scaling(4, 10, 10000.0, 2.0, &device, dtype).unwrap();
        assert_eq!(rope.base(), 20000.0); // base * scaling_factor
        
        let positions = Tensor::from_vec(vec![0u32, 1u32], (2,), &device).unwrap();
        let (cos, sin) = rope.get_cos_sin(&positions).unwrap();
        
        assert_eq!(cos.dims(), [2, 2]);
        assert_eq!(sin.dims(), [2, 2]);
    }
    
    #[test]
    fn test_expand_for_attention() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        
        let rope = RotaryEmbedding::new(4, 10, 10000.0, &device, dtype).unwrap();
        
        // Test 3D case
        let tensor_3d = Tensor::ones((2, 2), dtype, &device).unwrap(); // [seq_len, head_dim/2]
        let reference_3d = Tensor::zeros((2, 8, 4), dtype, &device).unwrap(); // [seq_len, num_heads, head_dim]
        
        let expanded = rope.expand_for_attention(&tensor_3d, &reference_3d).unwrap();
        assert_eq!(expanded.dims(), [2, 8, 2]);
        
        // Test 4D case
        let reference_4d = Tensor::zeros((1, 2, 8, 4), dtype, &device).unwrap(); // [batch, seq_len, num_heads, head_dim]
        let expanded = rope.expand_for_attention(&tensor_3d, &reference_4d).unwrap();
        assert_eq!(expanded.dims(), [1, 2, 8, 2]);
    }
    
    #[test]
    fn test_rotary_embedding_consistency() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        
        // Test that regular and optimized versions give same results
        let rope_regular = RotaryEmbedding::new(4, 10, 10000.0, &device, dtype).unwrap();
        let rope_optimized = OptimizedRotaryEmbedding::new(4, 10, 10000.0, &device, dtype).unwrap();
        
        let q = Tensor::randn(0f32, 1f32, (2, 4), &device).unwrap();
        let k = Tensor::randn(0f32, 1f32, (2, 4), &device).unwrap();
        let positions = Tensor::from_vec(vec![0u32, 1u32], (2,), &device).unwrap();
        
        let (q1, k1) = rope_regular.forward(&q, &k, &positions).unwrap();
        let (q2, k2) = rope_optimized.forward(&q, &k, &positions).unwrap();
        
        // Results should be very close
        let q_diff = (&q1 - &q2).unwrap().abs().unwrap().max_keepdim(0).unwrap();
        let k_diff = (&k1 - &k2).unwrap().abs().unwrap().max_keepdim(0).unwrap();
        
        let q_max_diff: f32 = q_diff.max_keepdim(1).unwrap().to_scalar().unwrap();
        let k_max_diff: f32 = k_diff.max_keepdim(1).unwrap().to_scalar().unwrap();
        
        assert!(q_max_diff < 1e-5, "Regular and optimized RoPE should give similar results for Q");
        assert!(k_max_diff < 1e-5, "Regular and optimized RoPE should give similar results for K");
    }
}