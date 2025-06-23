//! Linear layer implementations with tensor parallelism support
//! 
//! This module provides various linear layer implementations optimized
//! for distributed inference, including column and row parallel variants.

use candle_core::{Tensor, Device, DType, Result as CandleResult};
use candle_nn::{Linear, linear};
use std::sync::Arc;
use anyhow::Result;

/// Base trait for all linear layer implementations
pub trait LinearBase {
    /// Forward pass through the linear layer
    fn forward(&self, input: &Tensor) -> CandleResult<Tensor>;
    
    /// Get the input size of this layer
    fn input_size(&self) -> usize;
    
    /// Get the output size of this layer
    fn output_size(&self) -> usize;
    
    /// Load weights into this layer
    fn load_weight(&mut self, weight: Tensor) -> Result<()>;
}

/// Standard replicated linear layer (no tensor parallelism)
#[derive(Debug)]
pub struct ReplicatedLinear {
    linear: Linear,
    input_size: usize,
    output_size: usize,
}

impl ReplicatedLinear {
    /// Create a new replicated linear layer
    pub fn new(input_size: usize, output_size: usize, bias: bool, device: &Device, dtype: DType) -> CandleResult<Self> {
        let linear = linear(input_size, output_size, candle_nn::LinearConfig { bias }, device, dtype)?;
        
        Ok(Self {
            linear,
            input_size,
            output_size,
        })
    }
    
    /// Get reference to the underlying linear layer
    pub fn linear(&self) -> &Linear {
        &self.linear
    }
}

impl LinearBase for ReplicatedLinear {
    fn forward(&self, input: &Tensor) -> CandleResult<Tensor> {
        self.linear.forward(input)
    }
    
    fn input_size(&self) -> usize {
        self.input_size
    }
    
    fn output_size(&self) -> usize {
        self.output_size
    }
    
    fn load_weight(&mut self, weight: Tensor) -> Result<()> {
        // Verify dimensions match
        let expected_shape = [self.output_size, self.input_size];
        if weight.dims() != expected_shape {
            anyhow::bail!(
                "Weight shape mismatch: expected {:?}, got {:?}",
                expected_shape,
                weight.dims()
            );
        }
        
        // Replace the weight in the linear layer
        // Note: This is a simplified implementation - in practice, you'd need
        // to access the internal weight parameter of the Linear layer
        Ok(())
    }
}

/// Column-parallel linear layer for tensor parallelism
/// 
/// Splits the output dimension across multiple devices/processes.
/// Each device computes a portion of the output features.
#[derive(Debug)]
pub struct ColumnParallelLinear {
    linear: Linear,
    input_size: usize,
    output_size: usize,
    output_size_per_partition: usize,
    tp_rank: usize,
    tp_size: usize,
}

impl ColumnParallelLinear {
    /// Create a new column-parallel linear layer
    pub fn new(
        input_size: usize,
        output_size: usize,
        bias: bool,
        tp_rank: usize,
        tp_size: usize,
        device: &Device,
        dtype: DType,
    ) -> CandleResult<Self> {
        assert!(output_size % tp_size == 0, "Output size must be divisible by tensor parallel size");
        
        let output_size_per_partition = output_size / tp_size;
        let linear = linear(
            input_size,
            output_size_per_partition,
            candle_nn::LinearConfig { bias },
            device,
            dtype,
        )?;
        
        Ok(Self {
            linear,
            input_size,
            output_size,
            output_size_per_partition,
            tp_rank,
            tp_size,
        })
    }
    
    /// Get the partition size for this rank
    pub fn partition_size(&self) -> usize {
        self.output_size_per_partition
    }
    
    /// Get the tensor parallel rank
    pub fn tp_rank(&self) -> usize {
        self.tp_rank
    }
}

impl LinearBase for ColumnParallelLinear {
    fn forward(&self, input: &Tensor) -> CandleResult<Tensor> {
        // Each partition computes its portion of the output
        self.linear.forward(input)
    }
    
    fn input_size(&self) -> usize {
        self.input_size
    }
    
    fn output_size(&self) -> usize {
        self.output_size
    }
    
    fn load_weight(&mut self, weight: Tensor) -> Result<()> {
        // Extract the partition for this rank
        let start_idx = self.tp_rank * self.output_size_per_partition;
        let end_idx = start_idx + self.output_size_per_partition;
        
        let partition_weight = weight.narrow(0, start_idx, self.output_size_per_partition)?;
        
        // Verify dimensions
        let expected_shape = [self.output_size_per_partition, self.input_size];
        if partition_weight.dims() != expected_shape {
            anyhow::bail!(
                "Partition weight shape mismatch: expected {:?}, got {:?}",
                expected_shape,
                partition_weight.dims()
            );
        }
        
        Ok(())
    }
}

/// Row-parallel linear layer for tensor parallelism
/// 
/// Splits the input dimension across multiple devices/processes.
/// Requires all-reduce communication to combine results.
#[derive(Debug)]
pub struct RowParallelLinear {
    linear: Linear,
    input_size: usize,
    output_size: usize,
    input_size_per_partition: usize,
    tp_rank: usize,
    tp_size: usize,
}

impl RowParallelLinear {
    /// Create a new row-parallel linear layer
    pub fn new(
        input_size: usize,
        output_size: usize,
        bias: bool,
        tp_rank: usize,
        tp_size: usize,
        device: &Device,
        dtype: DType,
    ) -> CandleResult<Self> {
        assert!(input_size % tp_size == 0, "Input size must be divisible by tensor parallel size");
        
        let input_size_per_partition = input_size / tp_size;
        let linear = linear(
            input_size_per_partition,
            output_size,
            candle_nn::LinearConfig { bias: bias && tp_rank == 0 }, // Only rank 0 has bias
            device,
            dtype,
        )?;
        
        Ok(Self {
            linear,
            input_size,
            output_size,
            input_size_per_partition,
            tp_rank,
            tp_size,
        })
    }
    
    /// Get the partition size for this rank
    pub fn partition_size(&self) -> usize {
        self.input_size_per_partition
    }
}

impl LinearBase for RowParallelLinear {
    fn forward(&self, input: &Tensor) -> CandleResult<Tensor> {
        // Extract the input partition for this rank
        let start_idx = self.tp_rank * self.input_size_per_partition;
        let input_partition = input.narrow(input.dims().len() - 1, start_idx, self.input_size_per_partition)?;
        
        // Compute local result
        let local_output = self.linear.forward(&input_partition)?;
        
        // TODO: Add all-reduce communication here for multi-GPU setups
        // For now, return local result (works for single GPU)
        Ok(local_output)
    }
    
    fn input_size(&self) -> usize {
        self.input_size
    }
    
    fn output_size(&self) -> usize {
        self.output_size
    }
    
    fn load_weight(&mut self, weight: Tensor) -> Result<()> {
        // Extract the partition for this rank
        let start_idx = self.tp_rank * self.input_size_per_partition;
        let end_idx = start_idx + self.input_size_per_partition;
        
        let partition_weight = weight.narrow(1, start_idx, self.input_size_per_partition)?;
        
        // Verify dimensions
        let expected_shape = [self.output_size, self.input_size_per_partition];
        if partition_weight.dims() != expected_shape {
            anyhow::bail!(
                "Partition weight shape mismatch: expected {:?}, got {:?}",
                expected_shape,
                partition_weight.dims()
            );
        }
        
        Ok(())
    }
}

/// QKV parallel linear layer for attention
/// 
/// Combines query, key, and value projections into a single layer
/// with column parallelism support.
#[derive(Debug)]
pub struct QKVParallelLinear {
    linear: Linear,
    hidden_size: usize,
    head_size: usize,
    total_num_heads: usize,
    total_num_kv_heads: usize,
    num_heads: usize,
    num_kv_heads: usize,
    tp_rank: usize,
    tp_size: usize,
}

impl QKVParallelLinear {
    /// Create a new QKV parallel linear layer
    pub fn new(
        hidden_size: usize,
        head_size: usize,
        total_num_heads: usize,
        total_num_kv_heads: usize,
        bias: bool,
        tp_rank: usize,
        tp_size: usize,
        device: &Device,
        dtype: DType,
    ) -> CandleResult<Self> {
        assert!(total_num_heads % tp_size == 0, "Number of heads must be divisible by tensor parallel size");
        assert!(total_num_kv_heads % tp_size == 0, "Number of KV heads must be divisible by tensor parallel size");
        
        let num_heads = total_num_heads / tp_size;
        let num_kv_heads = total_num_kv_heads / tp_size;
        
        // Output size: Q + K + V projections
        let output_size = (num_heads + 2 * num_kv_heads) * head_size;
        
        let linear = linear(
            hidden_size,
            output_size,
            candle_nn::LinearConfig { bias },
            device,
            dtype,
        )?;
        
        Ok(Self {
            linear,
            hidden_size,
            head_size,
            total_num_heads,
            total_num_kv_heads,
            num_heads,
            num_kv_heads,
            tp_rank,
            tp_size,
        })
    }
    
    /// Split QKV output into separate Q, K, V tensors
    pub fn split_qkv(&self, qkv: &Tensor) -> CandleResult<(Tensor, Tensor, Tensor)> {
        let q_size = self.num_heads * self.head_size;
        let kv_size = self.num_kv_heads * self.head_size;
        
        let q = qkv.narrow(qkv.dims().len() - 1, 0, q_size)?;
        let k = qkv.narrow(qkv.dims().len() - 1, q_size, kv_size)?;
        let v = qkv.narrow(qkv.dims().len() - 1, q_size + kv_size, kv_size)?;
        
        Ok((q, k, v))
    }
    
    /// Get the number of heads per partition
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }
    
    /// Get the number of KV heads per partition
    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }
}

impl LinearBase for QKVParallelLinear {
    fn forward(&self, input: &Tensor) -> CandleResult<Tensor> {
        self.linear.forward(input)
    }
    
    fn input_size(&self) -> usize {
        self.hidden_size
    }
    
    fn output_size(&self) -> usize {
        (self.total_num_heads + 2 * self.total_num_kv_heads) * self.head_size
    }
    
    fn load_weight(&mut self, weight: Tensor) -> Result<()> {
        // This would need special handling for loading Q, K, V weights separately
        // and combining them into the packed format
        Ok(())
    }
}

/// Merged column-parallel linear layer
/// 
/// Combines multiple linear projections (e.g., gate and up projections in MLP)
/// into a single layer for efficiency.
#[derive(Debug)]
pub struct MergedColumnParallelLinear {
    linear: Linear,
    input_size: usize,
    output_sizes: Vec<usize>,
    total_output_size: usize,
    tp_rank: usize,
    tp_size: usize,
}

impl MergedColumnParallelLinear {
    /// Create a new merged column-parallel linear layer
    pub fn new(
        input_size: usize,
        output_sizes: Vec<usize>,
        bias: bool,
        tp_rank: usize,
        tp_size: usize,
        device: &Device,
        dtype: DType,
    ) -> CandleResult<Self> {
        let total_output_size: usize = output_sizes.iter().sum();
        assert!(total_output_size % tp_size == 0, "Total output size must be divisible by tensor parallel size");
        
        let output_size_per_partition = total_output_size / tp_size;
        let linear = linear(
            input_size,
            output_size_per_partition,
            candle_nn::LinearConfig { bias },
            device,
            dtype,
        )?;
        
        Ok(Self {
            linear,
            input_size,
            output_sizes,
            total_output_size,
            tp_rank,
            tp_size,
        })
    }
    
    /// Split the merged output into separate tensors
    pub fn split_output(&self, output: &Tensor) -> CandleResult<Vec<Tensor>> {
        let mut results = Vec::new();
        let mut start_idx = 0;
        
        for &size in &self.output_sizes {
            let partition_size = size / self.tp_size;
            let tensor = output.narrow(output.dims().len() - 1, start_idx, partition_size)?;
            results.push(tensor);
            start_idx += partition_size;
        }
        
        Ok(results)
    }
}

impl LinearBase for MergedColumnParallelLinear {
    fn forward(&self, input: &Tensor) -> CandleResult<Tensor> {
        self.linear.forward(input)
    }
    
    fn input_size(&self) -> usize {
        self.input_size
    }
    
    fn output_size(&self) -> usize {
        self.total_output_size
    }
    
    fn load_weight(&mut self, weight: Tensor) -> Result<()> {
        // This would need special handling for loading multiple weight matrices
        // and combining them into the merged format
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, DType};
    
    #[test]
    fn test_replicated_linear() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        
        let linear = ReplicatedLinear::new(128, 256, true, &device, dtype).unwrap();
        assert_eq!(linear.input_size(), 128);
        assert_eq!(linear.output_size(), 256);
        
        let input = Tensor::randn(0f32, 1f32, (4, 128), &device).unwrap();
        let output = linear.forward(&input).unwrap();
        assert_eq!(output.dims(), [4, 256]);
    }
    
    #[test]
    fn test_column_parallel_linear() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        
        let linear = ColumnParallelLinear::new(128, 256, false, 0, 2, &device, dtype).unwrap();
        assert_eq!(linear.input_size(), 128);
        assert_eq!(linear.output_size(), 256);
        assert_eq!(linear.partition_size(), 128); // 256 / 2
        
        let input = Tensor::randn(0f32, 1f32, (4, 128), &device).unwrap();
        let output = linear.forward(&input).unwrap();
        assert_eq!(output.dims(), [4, 128]); // Partition size
    }
    
    #[test]
    fn test_row_parallel_linear() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        
        let linear = RowParallelLinear::new(256, 128, false, 0, 2, &device, dtype).unwrap();
        assert_eq!(linear.input_size(), 256);
        assert_eq!(linear.output_size(), 128);
        assert_eq!(linear.partition_size(), 128); // 256 / 2
        
        let input = Tensor::randn(0f32, 1f32, (4, 256), &device).unwrap();
        let output = linear.forward(&input).unwrap();
        assert_eq!(output.dims(), [4, 128]);
    }
    
    #[test]
    fn test_qkv_parallel_linear() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        
        let qkv_linear = QKVParallelLinear::new(
            512, // hidden_size
            64,  // head_size
            8,   // total_num_heads
            8,   // total_num_kv_heads
            false,
            0,   // tp_rank
            1,   // tp_size
            &device,
            dtype,
        ).unwrap();
        
        assert_eq!(qkv_linear.num_heads(), 8);
        assert_eq!(qkv_linear.num_kv_heads(), 8);
        
        let input = Tensor::randn(0f32, 1f32, (4, 512), &device).unwrap();
        let qkv_output = qkv_linear.forward(&input).unwrap();
        
        // Output should be Q + K + V = 8*64 + 8*64 + 8*64 = 1536
        assert_eq!(qkv_output.dims(), [4, 1536]);
        
        let (q, k, v) = qkv_linear.split_qkv(&qkv_output).unwrap();
        assert_eq!(q.dims(), [4, 512]); // 8 heads * 64 head_size
        assert_eq!(k.dims(), [4, 512]);
        assert_eq!(v.dims(), [4, 512]);
    }
    
    #[test]
    fn test_merged_column_parallel_linear() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        
        let merged_linear = MergedColumnParallelLinear::new(
            256,
            vec![512, 512], // Two outputs of size 512 each
            false,
            0, // tp_rank
            1, // tp_size
            &device,
            dtype,
        ).unwrap();
        
        let input = Tensor::randn(0f32, 1f32, (4, 256), &device).unwrap();
        let output = merged_linear.forward(&input).unwrap();
        assert_eq!(output.dims(), [4, 1024]); // 512 + 512
        
        let split_outputs = merged_linear.split_output(&output).unwrap();
        assert_eq!(split_outputs.len(), 2);
        assert_eq!(split_outputs[0].dims(), [4, 512]);
        assert_eq!(split_outputs[1].dims(), [4, 512]);
    }
}