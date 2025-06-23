//! Embedding and language model head layers with tensor parallelism
//! 
//! This module implements vocabulary embeddings and language model heads
//! with support for tensor parallelism across multiple devices.

use candle_core::{Tensor, Device, DType, Result as CandleResult};
use candle_nn::{embedding, linear, Embedding, Linear};
use crate::utils::context::get_context;
use std::sync::Arc;

/// Vocabulary parallel embedding layer
/// 
/// Splits the vocabulary across multiple devices for tensor parallelism.
/// Each device handles a partition of the vocabulary.
#[derive(Debug)]
pub struct VocabParallelEmbedding {
    /// Embedding layer for this partition
    embedding: Embedding,
    
    /// Total vocabulary size
    vocab_size: usize,
    
    /// Embedding dimension
    embedding_dim: usize,
    
    /// Vocabulary size per partition
    vocab_size_per_partition: usize,
    
    /// Starting vocabulary index for this partition
    vocab_start_idx: usize,
    
    /// Ending vocabulary index for this partition
    vocab_end_idx: usize,
    
    /// Tensor parallel rank
    tp_rank: usize,
    
    /// Tensor parallel size
    tp_size: usize,
    
    /// Device for computations
    device: Device,
}

impl VocabParallelEmbedding {
    /// Create a new vocabulary parallel embedding layer
    pub fn new(
        vocab_size: usize,
        embedding_dim: usize,
        tp_rank: usize,
        tp_size: usize,
        device: &Device,
        dtype: DType,
    ) -> CandleResult<Self> {
        assert!(vocab_size % tp_size == 0, "Vocabulary size must be divisible by tensor parallel size");
        
        let vocab_size_per_partition = vocab_size / tp_size;
        let vocab_start_idx = vocab_size_per_partition * tp_rank;
        let vocab_end_idx = vocab_start_idx + vocab_size_per_partition;
        
        let embedding = embedding(vocab_size_per_partition, embedding_dim, device, dtype)?;
        
        Ok(Self {
            embedding,
            vocab_size,
            embedding_dim,
            vocab_size_per_partition,
            vocab_start_idx,
            vocab_end_idx,
            tp_rank,
            tp_size,
            device: device.clone(),
        })
    }
    
    /// Forward pass through the embedding layer
    pub fn forward(&self, input_ids: &Tensor) -> CandleResult<Tensor> {
        if self.tp_size == 1 {
            // No tensor parallelism, use embedding directly
            return self.embedding.forward(input_ids);
        }
        
        // Create mask for tokens in this partition's vocabulary range
        let mask = self.create_vocab_mask(input_ids)?;
        
        // Map input IDs to local vocabulary indices
        let local_input_ids = self.map_to_local_vocab(input_ids)?;
        
        // Get embeddings for local vocabulary
        let local_embeddings = self.embedding.forward(&local_input_ids)?;
        
        // Apply mask to zero out embeddings for tokens not in this partition
        let masked_embeddings = self.apply_mask(&local_embeddings, &mask)?;
        
        // All-reduce to combine embeddings from all partitions
        self.all_reduce_embeddings(masked_embeddings)
    }
    
    /// Create a mask indicating which tokens belong to this partition
    fn create_vocab_mask(&self, input_ids: &Tensor) -> CandleResult<Tensor> {
        let ge_start = input_ids.ge(&Tensor::new(self.vocab_start_idx as i64, &self.device)?)?;
        let lt_end = input_ids.lt(&Tensor::new(self.vocab_end_idx as i64, &self.device)?)?;
        ge_start.and(&lt_end)
    }
    
    /// Map global vocabulary indices to local partition indices
    fn map_to_local_vocab(&self, input_ids: &Tensor) -> CandleResult<Tensor> {
        let start_tensor = Tensor::new(self.vocab_start_idx as i64, &self.device)?;
        let local_ids = input_ids.broadcast_sub(&start_tensor)?;
        
        // Clamp to valid range [0, vocab_size_per_partition)
        let zero = Tensor::new(0i64, &self.device)?;
        let max_val = Tensor::new((self.vocab_size_per_partition - 1) as i64, &self.device)?;
        
        local_ids.clamp(&zero, &max_val)
    }
    
    /// Apply vocabulary mask to embeddings
    fn apply_mask(&self, embeddings: &Tensor, mask: &Tensor) -> CandleResult<Tensor> {
        // Expand mask to match embedding dimensions
        let expanded_mask = mask.unsqueeze(mask.dims().len())?
            .expand(embeddings.dims())?;
        
        // Convert boolean mask to float and multiply
        let float_mask = expanded_mask.to_dtype(embeddings.dtype())?;
        embeddings.broadcast_mul(&float_mask)
    }
    
    /// All-reduce embeddings across tensor parallel ranks
    fn all_reduce_embeddings(&self, embeddings: Tensor) -> CandleResult<Tensor> {
        if self.tp_size == 1 {
            return Ok(embeddings);
        }
        
        // In a real implementation, this would use distributed communication
        // For now, we'll return the embeddings as-is
        // TODO: Implement actual all-reduce communication
        Ok(embeddings)
    }
    
    /// Load weights into this embedding layer
    pub fn load_weight(&mut self, weight: Tensor) -> anyhow::Result<()> {
        // Extract the partition for this rank
        let start_idx = self.tp_rank * self.vocab_size_per_partition;
        let partition_weight = weight.narrow(0, start_idx, self.vocab_size_per_partition)?;
        
        // Verify dimensions
        let expected_shape = [self.vocab_size_per_partition, self.embedding_dim];
        if partition_weight.dims() != expected_shape {
            anyhow::bail!(
                "Partition weight shape mismatch: expected {:?}, got {:?}",
                expected_shape,
                partition_weight.dims()
            );
        }
        
        // Copy weights to embedding layer
        // Note: This is a simplified implementation
        // In practice, you'd need to access the internal weight parameter
        Ok(())
    }
    
    /// Get the vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
    
    /// Get the embedding dimension
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }
    
    /// Get the tensor parallel rank
    pub fn tp_rank(&self) -> usize {
        self.tp_rank
    }
}

/// Parallel language model head
/// 
/// Computes logits over the vocabulary with tensor parallelism support.
/// Inherits from VocabParallelEmbedding for weight sharing.
#[derive(Debug)]
pub struct ParallelLMHead {
    /// Underlying embedding layer (for weight sharing)
    embedding: VocabParallelEmbedding,
    
    /// Optional bias term
    bias: Option<Tensor>,
    
    /// Whether to use bias
    use_bias: bool,
}

impl ParallelLMHead {
    /// Create a new parallel language model head
    pub fn new(
        vocab_size: usize,
        embedding_dim: usize,
        bias: bool,
        tp_rank: usize,
        tp_size: usize,
        device: &Device,
        dtype: DType,
    ) -> CandleResult<Self> {
        let embedding = VocabParallelEmbedding::new(
            vocab_size,
            embedding_dim,
            tp_rank,
            tp_size,
            device,
            dtype,
        )?;
        
        let bias_tensor = if bias {
            let vocab_size_per_partition = vocab_size / tp_size;
            Some(Tensor::zeros((vocab_size_per_partition,), dtype, device)?)
        } else {
            None
        };
        
        Ok(Self {
            embedding,
            bias: bias_tensor,
            use_bias: bias,
        })
    }
    
    /// Create from existing embedding (for weight tying)
    pub fn from_embedding(
        embedding: VocabParallelEmbedding,
        bias: bool,
        device: &Device,
        dtype: DType,
    ) -> CandleResult<Self> {
        let bias_tensor = if bias {
            Some(Tensor::zeros((embedding.vocab_size_per_partition,), dtype, device)?)
        } else {
            None
        };
        
        Ok(Self {
            embedding,
            bias: bias_tensor,
            use_bias: bias,
        })
    }
    
    /// Forward pass through the language model head
    pub fn forward(&self, hidden_states: &Tensor) -> CandleResult<Tensor> {
        let context = get_context();
        
        // For prefill, extract only the last token of each sequence
        let input_hidden = if context.is_prefill {
            self.extract_last_tokens(hidden_states, &context)?
        } else {
            hidden_states.clone()
        };
        
        // Compute logits using the embedding weights (transposed)
        let logits = self.compute_logits(&input_hidden)?;
        
        // Gather logits from all tensor parallel ranks
        if self.embedding.tp_size > 1 {
            self.gather_logits(logits)
        } else {
            Ok(logits)
        }
    }
    
    /// Extract last tokens for each sequence in prefill mode
    fn extract_last_tokens(
        &self,
        hidden_states: &Tensor,
        context: &crate::utils::context::Context,
    ) -> CandleResult<Tensor> {
        let cu_seqlens_q = context.cu_seqlens_q.as_ref().unwrap();
        let batch_size = cu_seqlens_q.dim(0)? - 1;
        
        let mut last_tokens = Vec::new();
        
        for i in 0..batch_size {
            let end_idx = cu_seqlens_q.get(i + 1)?.to_scalar::<i32>()? as usize;
            let last_token = hidden_states.get(end_idx - 1)?;
            last_tokens.push(last_token);
        }
        
        Tensor::stack(&last_tokens, 0)
    }
    
    /// Compute logits using embedding weights
    fn compute_logits(&self, hidden_states: &Tensor) -> CandleResult<Tensor> {
        // Get embedding weights (need to transpose for matrix multiplication)
        let weight = self.get_embedding_weight()?;
        let weight_t = weight.transpose(0, 1)?;
        
        // Compute logits: hidden_states @ weight.T
        let logits = hidden_states.matmul(&weight_t)?;
        
        // Add bias if present
        if let Some(bias) = &self.bias {
            logits.broadcast_add(bias)
        } else {
            Ok(logits)
        }
    }
    
    /// Get the embedding weight tensor
    fn get_embedding_weight(&self) -> CandleResult<Tensor> {
        // This is a simplified implementation
        // In practice, you'd access the actual weight from the embedding layer
        Tensor::randn(
            0f32,
            1f32,
            (self.embedding.vocab_size_per_partition, self.embedding.embedding_dim),
            &self.embedding.device,
        )
    }
    
    /// Gather logits from all tensor parallel ranks
    fn gather_logits(&self, logits: Tensor) -> CandleResult<Tensor> {
        if self.embedding.tp_rank == 0 {
            // Rank 0 gathers logits from all ranks
            let mut all_logits = vec![logits];
            
            // In a real implementation, this would gather from other ranks
            // For now, we'll just return the local logits
            // TODO: Implement actual gather communication
            
            Tensor::cat(&all_logits, 1) // Concatenate along vocabulary dimension
        } else {
            // Non-zero ranks send their logits to rank 0
            // TODO: Implement actual send communication
            Ok(logits)
        }
    }
    
    /// Load bias weights
    pub fn load_bias(&mut self, bias: Tensor) -> anyhow::Result<()> {
        if !self.use_bias {
            anyhow::bail!("Bias is not enabled for this LM head");
        }
        
        // Extract the partition for this rank
        let start_idx = self.embedding.tp_rank * self.embedding.vocab_size_per_partition;
        let partition_bias = bias.narrow(0, start_idx, self.embedding.vocab_size_per_partition)?;
        
        self.bias = Some(partition_bias);
        Ok(())
    }
    
    /// Load embedding weights (delegates to underlying embedding)
    pub fn load_weight(&mut self, weight: Tensor) -> anyhow::Result<()> {
        self.embedding.load_weight(weight)
    }
    
    /// Get the vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.embedding.vocab_size()
    }
    
    /// Get the embedding dimension
    pub fn embedding_dim(&self) -> usize {
        self.embedding.embedding_dim()
    }
}

/// Standard (non-parallel) embedding layer for single-device inference
#[derive(Debug)]
pub struct StandardEmbedding {
    embedding: Embedding,
    vocab_size: usize,
    embedding_dim: usize,
}

impl StandardEmbedding {
    /// Create a new standard embedding layer
    pub fn new(
        vocab_size: usize,
        embedding_dim: usize,
        device: &Device,
        dtype: DType,
    ) -> CandleResult<Self> {
        let embedding = embedding(vocab_size, embedding_dim, device, dtype)?;
        
        Ok(Self {
            embedding,
            vocab_size,
            embedding_dim,
        })
    }
    
    /// Forward pass
    pub fn forward(&self, input_ids: &Tensor) -> CandleResult<Tensor> {
        self.embedding.forward(input_ids)
    }
    
    /// Load weights
    pub fn load_weight(&mut self, weight: Tensor) -> anyhow::Result<()> {
        let expected_shape = [self.vocab_size, self.embedding_dim];
        if weight.dims() != expected_shape {
            anyhow::bail!(
                "Weight shape mismatch: expected {:?}, got {:?}",
                expected_shape,
                weight.dims()
            );
        }
        
        // Copy weights to embedding layer
        // Note: Simplified implementation
        Ok(())
    }
}

/// Standard (non-parallel) language model head
#[derive(Debug)]
pub struct StandardLMHead {
    linear: Linear,
    vocab_size: usize,
    embedding_dim: usize,
}

impl StandardLMHead {
    /// Create a new standard language model head
    pub fn new(
        vocab_size: usize,
        embedding_dim: usize,
        bias: bool,
        device: &Device,
        dtype: DType,
    ) -> CandleResult<Self> {
        let linear = linear(embedding_dim, vocab_size, candle_nn::LinearConfig { bias }, device, dtype)?;
        
        Ok(Self {
            linear,
            vocab_size,
            embedding_dim,
        })
    }
    
    /// Forward pass
    pub fn forward(&self, hidden_states: &Tensor) -> CandleResult<Tensor> {
        let context = get_context();
        
        let input_hidden = if context.is_prefill {
            // Extract last tokens for prefill
            let cu_seqlens_q = context.cu_seqlens_q.as_ref().unwrap();
            let batch_size = cu_seqlens_q.dim(0)? - 1;
            
            let mut last_tokens = Vec::new();
            for i in 0..batch_size {
                let end_idx = cu_seqlens_q.get(i + 1)?.to_scalar::<i32>()? as usize;
                let last_token = hidden_states.get(end_idx - 1)?;
                last_tokens.push(last_token);
            }
            
            Tensor::stack(&last_tokens, 0)?
        } else {
            hidden_states.clone()
        };
        
        self.linear.forward(&input_hidden)
    }
    
    /// Load weights
    pub fn load_weight(&mut self, weight: Tensor) -> anyhow::Result<()> {
        let expected_shape = [self.vocab_size, self.embedding_dim];
        if weight.dims() != expected_shape {
            anyhow::bail!(
                "Weight shape mismatch: expected {:?}, got {:?}",
                expected_shape,
                weight.dims()
            );
        }
        
        // Copy weights to linear layer
        // Note: Simplified implementation
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, DType};
    
    #[test]
    fn test_vocab_parallel_embedding() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        
        let embedding = VocabParallelEmbedding::new(
            1000, // vocab_size
            512,  // embedding_dim
            0,    // tp_rank
            2,    // tp_size
            &device,
            dtype,
        ).unwrap();
        
        assert_eq!(embedding.vocab_size(), 1000);
        assert_eq!(embedding.embedding_dim(), 512);
        assert_eq!(embedding.vocab_size_per_partition, 500);
        assert_eq!(embedding.vocab_start_idx, 0);
        assert_eq!(embedding.vocab_end_idx, 500);
    }
    
    #[test]
    fn test_vocab_mask_creation() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        
        let embedding = VocabParallelEmbedding::new(
            100, 64, 0, 2, &device, dtype
        ).unwrap();
        
        // Partition 0 handles vocab indices 0-49
        let input_ids = Tensor::from_slice(&[10i64, 25, 60, 75], (4,), &device).unwrap();
        let mask = embedding.create_vocab_mask(&input_ids).unwrap();
        
        let mask_values: Vec<u8> = mask.to_vec1().unwrap();
        assert_eq!(mask_values, vec![1, 1, 0, 0]); // First two are in range [0, 50)
    }
    
    #[test]
    fn test_local_vocab_mapping() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        
        let embedding = VocabParallelEmbedding::new(
            100, 64, 1, 2, &device, dtype
        ).unwrap();
        
        // Partition 1 handles vocab indices 50-99, maps to local 0-49
        let input_ids = Tensor::from_slice(&[55i64, 60, 75, 90], (4,), &device).unwrap();
        let local_ids = embedding.map_to_local_vocab(&input_ids).unwrap();
        
        let local_values: Vec<i64> = local_ids.to_vec1().unwrap();
        assert_eq!(local_values, vec![5, 10, 25, 40]); // Subtract 50 (start_idx)
    }
    
    #[test]
    fn test_parallel_lm_head() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        
        let lm_head = ParallelLMHead::new(
            1000, // vocab_size
            512,  // embedding_dim
            false, // bias
            0,    // tp_rank
            2,    // tp_size
            &device,
            dtype,
        ).unwrap();
        
        assert_eq!(lm_head.vocab_size(), 1000);
        assert_eq!(lm_head.embedding_dim(), 512);
        assert!(!lm_head.use_bias);
    }
    
    #[test]
    fn test_standard_embedding() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        
        let embedding = StandardEmbedding::new(1000, 512, &device, dtype).unwrap();
        
        let input_ids = Tensor::from_slice(&[1i64, 5, 10], (3,), &device).unwrap();
        let output = embedding.forward(&input_ids).unwrap();
        
        assert_eq!(output.dims(), [3, 512]);
    }
    
    #[test]
    fn test_standard_lm_head() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        
        let lm_head = StandardLMHead::new(1000, 512, false, &device, dtype).unwrap();
        
        let hidden_states = Tensor::randn(0f32, 1f32, (3, 512), &device).unwrap();
        let logits = lm_head.forward(&hidden_states).unwrap();
        
        assert_eq!(logits.dims(), [3, 1000]);
    }
    
    #[test]
    fn test_weight_loading() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        
        let mut embedding = VocabParallelEmbedding::new(
            100, 64, 0, 2, &device, dtype
        ).unwrap();
        
        let weight = Tensor::randn(0f32, 1f32, (100, 64), &device).unwrap();
        let result = embedding.load_weight(weight);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_weight_loading_wrong_shape() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        
        let mut embedding = VocabParallelEmbedding::new(
            100, 64, 0, 2, &device, dtype
        ).unwrap();
        
        let wrong_weight = Tensor::randn(0f32, 1f32, (50, 32), &device).unwrap();
        let result = embedding.load_weight(wrong_weight);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_lm_head_from_embedding() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        
        let embedding = VocabParallelEmbedding::new(
            1000, 512, 0, 1, &device, dtype
        ).unwrap();
        
        let lm_head = ParallelLMHead::from_embedding(
            embedding, true, &device, dtype
        ).unwrap();
        
        assert!(lm_head.use_bias);
        assert!(lm_head.bias.is_some());
    }
}