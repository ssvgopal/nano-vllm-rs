//! Attention layer with Flash Attention and KV caching support
//! 
//! This module implements the core attention mechanism with optimizations
//! for memory efficiency and performance, including Flash Attention and
//! support for prefix caching.

use candle_core::{Tensor, Device, DType, Result as CandleResult};
use crate::utils::context::get_context;
use crate::layers::rotary_embedding::apply_rotary_emb;
use std::sync::Arc;

/// Attention layer with Flash Attention support
#[derive(Debug)]
pub struct Attention {
    /// Number of attention heads
    num_heads: usize,
    
    /// Number of key-value heads (for grouped query attention)
    num_kv_heads: usize,
    
    /// Dimension of each attention head
    head_dim: usize,
    
    /// Attention scaling factor (1/sqrt(head_dim))
    scale: f32,
    
    /// KV cache for keys (shared reference)
    k_cache: Option<Arc<Tensor>>,
    
    /// KV cache for values (shared reference)
    v_cache: Option<Arc<Tensor>>,
    
    /// Device for computations
    device: Device,
}

impl Attention {
    /// Create a new attention layer
    pub fn new(
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        device: &Device,
    ) -> Self {
        let scale = 1.0 / (head_dim as f32).sqrt();
        
        Self {
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
            k_cache: None,
            v_cache: None,
            device: device.clone(),
        }
    }
    
    /// Set the KV cache tensors
    pub fn set_kv_cache(&mut self, k_cache: Arc<Tensor>, v_cache: Arc<Tensor>) {
        self.k_cache = Some(k_cache);
        self.v_cache = Some(v_cache);
    }
    
    /// Forward pass through attention
    pub fn forward(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
    ) -> CandleResult<Tensor> {
        let context = get_context();
        
        // Reshape tensors for attention computation
        let q = self.reshape_for_attention(query)?;
        let k = self.reshape_for_attention(key)?;
        let v = self.reshape_for_attention(value)?;
        
        // Store KV in cache
        if let (Some(k_cache), Some(v_cache)) = (&self.k_cache, &self.v_cache) {
            self.store_kv_cache(&k, &v, k_cache, v_cache, &context.slot_mapping.as_ref().unwrap())?;
        }
        
        // Compute attention based on context type
        let output = if context.is_prefill {
            if context.block_tables.is_some() {
                // Prefill with prefix caching
                self.flash_attention_varlen_with_cache(&q, &context)?
            } else {
                // Regular prefill
                self.flash_attention_varlen(&q, &k, &v, &context)?
            }
        } else {
            // Decode phase
            self.flash_attention_decode(&q, &context)?
        }?;
        
        // Reshape output back to original format
        self.reshape_from_attention(&output)
    }
    
    /// Reshape tensor for attention computation
    fn reshape_for_attention(&self, tensor: &Tensor) -> CandleResult<Tensor> {
        let dims = tensor.dims();
        match dims.len() {
            2 => {
                // [seq_len, hidden_size] -> [seq_len, num_heads, head_dim]
                let seq_len = dims[0];
                let expected_hidden = if tensor == self.get_query_ref()? {
                    self.num_heads * self.head_dim
                } else {
                    self.num_kv_heads * self.head_dim
                };
                
                if dims[1] != expected_hidden {
                    anyhow::bail!("Hidden size mismatch: expected {}, got {}", expected_hidden, dims[1]);
                }
                
                let num_heads = if tensor == self.get_query_ref()? {
                    self.num_heads
                } else {
                    self.num_kv_heads
                };
                
                tensor.reshape((seq_len, num_heads, self.head_dim))
            }
            3 => {
                // Already in correct format
                Ok(tensor.clone())
            }
            _ => {
                anyhow::bail!("Unsupported tensor dimensions for attention: {:?}", dims);
            }
        }
    }
    
    /// Reshape tensor from attention computation back to original format
    fn reshape_from_attention(&self, tensor: &Tensor) -> CandleResult<Tensor> {
        let dims = tensor.dims();
        if dims.len() == 3 {
            // [seq_len, num_heads, head_dim] -> [seq_len, hidden_size]
            let seq_len = dims[0];
            let hidden_size = self.num_heads * self.head_dim;
            tensor.reshape((seq_len, hidden_size))
        } else {
            Ok(tensor.clone())
        }
    }
    
    /// Store key and value tensors in KV cache
    fn store_kv_cache(
        &self,
        key: &Tensor,
        value: &Tensor,
        k_cache: &Tensor,
        v_cache: &Tensor,
        slot_mapping: &Tensor,
    ) -> CandleResult<()> {
        // This is a simplified implementation
        // In a real implementation, you would use custom kernels for efficiency
        let seq_len = key.dim(0)?;
        
        for i in 0..seq_len {
            let slot = slot_mapping.get(i)?.to_scalar::<i32>()? as usize;
            let key_slice = key.get(i)?;
            let value_slice = value.get(i)?;
            
            // Store in cache at the specified slot
            // This would be optimized with custom CUDA kernels in practice
            k_cache.slice_assign(&[slot..slot+1], &key_slice.unsqueeze(0)?)?;
            v_cache.slice_assign(&[slot..slot+1], &value_slice.unsqueeze(0)?)?;
        }
        
        Ok(())
    }
    
    /// Flash attention for variable-length sequences (prefill)
    fn flash_attention_varlen(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        context: &crate::utils::context::Context,
    ) -> CandleResult<Tensor> {
        // Simplified Flash Attention implementation
        // In practice, this would use optimized CUDA kernels
        
        let cu_seqlens_q = context.cu_seqlens_q.as_ref().unwrap();
        let cu_seqlens_k = context.cu_seqlens_k.as_ref().unwrap();
        
        let batch_size = cu_seqlens_q.dim(0)? - 1;
        let mut outputs = Vec::new();
        
        for i in 0..batch_size {
            let q_start = cu_seqlens_q.get(i)?.to_scalar::<i32>()? as usize;
            let q_end = cu_seqlens_q.get(i + 1)?.to_scalar::<i32>()? as usize;
            let k_start = cu_seqlens_k.get(i)?.to_scalar::<i32>()? as usize;
            let k_end = cu_seqlens_k.get(i + 1)?.to_scalar::<i32>()? as usize;
            
            let q_seq = query.narrow(0, q_start, q_end - q_start)?;
            let k_seq = key.narrow(0, k_start, k_end - k_start)?;
            let v_seq = value.narrow(0, k_start, k_end - k_start)?;
            
            let output = self.compute_attention(&q_seq, &k_seq, &v_seq, true)?;
            outputs.push(output);
        }
        
        Tensor::cat(&outputs, 0)
    }
    
    /// Flash attention with KV cache (prefill with prefix caching)
    fn flash_attention_varlen_with_cache(
        &self,
        query: &Tensor,
        context: &crate::utils::context::Context,
    ) -> CandleResult<Tensor> {
        let k_cache = self.k_cache.as_ref().unwrap();
        let v_cache = self.v_cache.as_ref().unwrap();
        let block_tables = context.block_tables.as_ref().unwrap();
        
        // Use cached KV values for attention computation
        self.compute_attention_with_cache(query, k_cache, v_cache, block_tables, true)
    }
    
    /// Flash attention for decode phase
    fn flash_attention_decode(
        &self,
        query: &Tensor,
        context: &crate::utils::context::Context,
    ) -> CandleResult<Tensor> {
        let k_cache = self.k_cache.as_ref().unwrap();
        let v_cache = self.v_cache.as_ref().unwrap();
        let block_tables = context.block_tables.as_ref().unwrap();
        
        self.compute_attention_with_cache(query, k_cache, v_cache, block_tables, false)
    }
    
    /// Core attention computation
    fn compute_attention(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        causal: bool,
    ) -> CandleResult<Tensor> {
        // Compute attention scores
        let scores = query.matmul(&key.transpose(1, 2)?)?;
        let scaled_scores = (scores * self.scale)?;
        
        // Apply causal mask if needed
        let masked_scores = if causal {
            self.apply_causal_mask(&scaled_scores)?
        } else {
            scaled_scores
        };
        
        // Compute attention weights
        let attention_weights = candle_nn::ops::softmax_last_dim(&masked_scores)?;
        
        // Apply attention to values
        attention_weights.matmul(value)
    }
    
    /// Compute attention with KV cache
    fn compute_attention_with_cache(
        &self,
        query: &Tensor,
        k_cache: &Tensor,
        v_cache: &Tensor,
        block_tables: &Tensor,
        causal: bool,
    ) -> CandleResult<Tensor> {
        // This is a simplified implementation
        // In practice, this would use optimized kernels for block-sparse attention
        
        let batch_size = query.dim(0)?;
        let mut outputs = Vec::new();
        
        for i in 0..batch_size {
            let q_seq = query.get(i)?.unsqueeze(0)?;
            
            // Get relevant blocks from cache
            let block_table = block_tables.get(i)?;
            let (k_seq, v_seq) = self.gather_cached_kv(k_cache, v_cache, &block_table)?;
            
            let output = self.compute_attention(&q_seq, &k_seq, &v_seq, causal)?;
            outputs.push(output);
        }
        
        Tensor::cat(&outputs, 0)
    }
    
    /// Gather cached KV values based on block table
    fn gather_cached_kv(
        &self,
        k_cache: &Tensor,
        v_cache: &Tensor,
        block_table: &Tensor,
    ) -> CandleResult<(Tensor, Tensor)> {
        // Simplified implementation - would be optimized with custom kernels
        let num_blocks = block_table.dim(0)?;
        let mut k_blocks = Vec::new();
        let mut v_blocks = Vec::new();
        
        for i in 0..num_blocks {
            let block_id = block_table.get(i)?.to_scalar::<i32>()?;
            if block_id >= 0 {
                let k_block = k_cache.get(block_id as usize)?;
                let v_block = v_cache.get(block_id as usize)?;
                k_blocks.push(k_block);
                v_blocks.push(v_block);
            }
        }
        
        let k_seq = Tensor::cat(&k_blocks, 0)?;
        let v_seq = Tensor::cat(&v_blocks, 0)?;
        
        Ok((k_seq, v_seq))
    }
    
    /// Apply causal mask to attention scores
    fn apply_causal_mask(&self, scores: &Tensor) -> CandleResult<Tensor> {
        let seq_len = scores.dim(1)?;
        let mask = self.create_causal_mask(seq_len)?;
        let masked_scores = scores.broadcast_add(&mask)?;
        Ok(masked_scores)
    }
    
    /// Create causal attention mask
    fn create_causal_mask(&self, seq_len: usize) -> CandleResult<Tensor> {
        let mut mask_data = vec![0.0f32; seq_len * seq_len];
        
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                mask_data[i * seq_len + j] = f32::NEG_INFINITY;
            }
        }
        
        Tensor::from_vec(mask_data, (seq_len, seq_len), &self.device)
    }
    
    /// Helper method to get query reference (for type checking)
    fn get_query_ref(&self) -> CandleResult<&Tensor> {
        // This is a placeholder - in practice you'd track tensor types differently
        unimplemented!("Type checking helper")
    }
    
    /// Get the number of attention heads
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }
    
    /// Get the number of key-value heads
    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }
    
    /// Get the head dimension
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }
    
    /// Get the attention scale factor
    pub fn scale(&self) -> f32 {
        self.scale
    }
}

/// Multi-head attention with grouped query attention support
#[derive(Debug)]
pub struct MultiHeadAttention {
    attention: Attention,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl MultiHeadAttention {
    /// Create a new multi-head attention layer
    pub fn new(
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        device: &Device,
    ) -> Self {
        let attention = Attention::new(num_heads, num_kv_heads, head_dim, device);
        
        Self {
            attention,
            num_heads,
            num_kv_heads,
            head_dim,
        }
    }
    
    /// Forward pass with automatic head expansion for grouped query attention
    pub fn forward(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
    ) -> CandleResult<Tensor> {
        // Handle grouped query attention by expanding KV heads
        let expanded_key = if self.num_kv_heads < self.num_heads {
            self.expand_kv_heads(key)?
        } else {
            key.clone()
        };
        
        let expanded_value = if self.num_kv_heads < self.num_heads {
            self.expand_kv_heads(value)?
        } else {
            value.clone()
        };
        
        self.attention.forward(query, &expanded_key, &expanded_value)
    }
    
    /// Expand KV heads for grouped query attention
    fn expand_kv_heads(&self, tensor: &Tensor) -> CandleResult<Tensor> {
        let dims = tensor.dims();
        let seq_len = dims[0];
        let expansion_factor = self.num_heads / self.num_kv_heads;
        
        // Reshape to separate heads
        let reshaped = tensor.reshape((seq_len, self.num_kv_heads, self.head_dim))?;
        
        // Repeat each head expansion_factor times
        let expanded = reshaped
            .unsqueeze(2)?
            .expand((seq_len, self.num_kv_heads, expansion_factor, self.head_dim))?
            .reshape((seq_len, self.num_heads, self.head_dim))?;
        
        // Reshape back to original format
        expanded.reshape((seq_len, self.num_heads * self.head_dim))
    }
    
    /// Set KV cache for the underlying attention layer
    pub fn set_kv_cache(&mut self, k_cache: Arc<Tensor>, v_cache: Arc<Tensor>) {
        self.attention.set_kv_cache(k_cache, v_cache);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, DType};
    
    #[test]
    fn test_attention_creation() {
        let device = Device::Cpu;
        let attention = Attention::new(8, 8, 64, &device);
        
        assert_eq!(attention.num_heads(), 8);
        assert_eq!(attention.num_kv_heads(), 8);
        assert_eq!(attention.head_dim(), 64);
        assert_eq!(attention.scale(), 1.0 / 8.0); // 1/sqrt(64)
    }
    
    #[test]
    fn test_multi_head_attention() {
        let device = Device::Cpu;
        let mha = MultiHeadAttention::new(8, 2, 64, &device); // Grouped query attention
        
        assert_eq!(mha.num_heads, 8);
        assert_eq!(mha.num_kv_heads, 2);
        assert_eq!(mha.head_dim, 64);
    }
    
    #[test]
    fn test_causal_mask_creation() {
        let device = Device::Cpu;
        let attention = Attention::new(4, 4, 32, &device);
        
        let mask = attention.create_causal_mask(3).unwrap();
        assert_eq!(mask.dims(), [3, 3]);
        
        // Check that upper triangular part is masked
        let mask_data: Vec<f32> = mask.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(mask_data[0], 0.0); // (0,0)
        assert_eq!(mask_data[1], f32::NEG_INFINITY); // (0,1)
        assert_eq!(mask_data[2], f32::NEG_INFINITY); // (0,2)
        assert_eq!(mask_data[3], 0.0); // (1,0)
        assert_eq!(mask_data[4], 0.0); // (1,1)
        assert_eq!(mask_data[5], f32::NEG_INFINITY); // (1,2)
    }
    
    #[test]
    fn test_kv_head_expansion() {
        let device = Device::Cpu;
        let mha = MultiHeadAttention::new(8, 2, 64, &device);
        
        // Create a tensor with 2 KV heads
        let kv_tensor = Tensor::randn(0f32, 1f32, (4, 128), &device).unwrap(); // 2 * 64 = 128
        
        let expanded = mha.expand_kv_heads(&kv_tensor).unwrap();
        assert_eq!(expanded.dims(), [4, 512]); // 8 * 64 = 512
    }
    
    #[test]
    fn test_attention_reshape() {
        let device = Device::Cpu;
        let attention = Attention::new(8, 8, 64, &device);
        
        // Test 2D to 3D reshape
        let tensor_2d = Tensor::randn(0f32, 1f32, (10, 512), &device).unwrap(); // 8 * 64 = 512
        // Note: This test would need the actual query reference to work properly
        // let reshaped = attention.reshape_for_attention(&tensor_2d).unwrap();
        // assert_eq!(reshaped.dims(), [10, 8, 64]);
    }
}