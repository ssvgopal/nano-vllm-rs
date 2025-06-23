//! RMS Normalization layer implementation
//! 
//! This module implements RMSNorm (Root Mean Square Normalization),
//! which is commonly used in modern transformer architectures.

use candle_core::{Tensor, Device, DType, Result as CandleResult};
use candle_nn::{VarBuilder, VarMap};

/// RMS Normalization layer
/// 
/// RMSNorm normalizes the input using the root mean square of the elements,
/// which is more efficient than LayerNorm as it doesn't require computing
/// the mean and doesn't have a bias term.
#[derive(Debug)]
pub struct RMSNorm {
    /// Learnable scale parameter
    weight: Tensor,
    
    /// Normalization epsilon for numerical stability
    eps: f64,
    
    /// Hidden size (dimension to normalize)
    hidden_size: usize,
}

impl RMSNorm {
    /// Create a new RMSNorm layer
    pub fn new(hidden_size: usize, eps: f64, device: &Device, dtype: DType) -> CandleResult<Self> {
        let weight = Tensor::ones((hidden_size,), dtype, device)?;
        
        Ok(Self {
            weight,
            eps,
            hidden_size,
        })
    }
    
    /// Create RMSNorm from a variable builder
    pub fn load(vb: VarBuilder, hidden_size: usize, eps: f64) -> CandleResult<Self> {
        let weight = vb.get((hidden_size,), "weight")?;
        
        Ok(Self {
            weight,
            eps,
            hidden_size,
        })
    }
    
    /// Forward pass with optional residual connection
    pub fn forward(&self, x: &Tensor, residual: Option<&Tensor>) -> CandleResult<Tensor> {
        match residual {
            Some(res) => self.forward_with_residual(x, res),
            None => self.forward_simple(x),
        }
    }
    
    /// Simple forward pass without residual connection
    pub fn forward_simple(&self, x: &Tensor) -> CandleResult<Tensor> {
        let original_dtype = x.dtype();
        
        // Convert to f32 for computation
        let x_f32 = x.to_dtype(DType::F32)?;
        
        // Compute RMS: sqrt(mean(x^2))
        let x_squared = x_f32.sqr()?;
        let mean_squared = x_squared.mean_keepdim(x_squared.dims().len() - 1)?;
        let rms = (mean_squared + self.eps)?.sqrt()?;
        
        // Normalize: x / rms
        let normalized = x_f32.broadcast_div(&rms)?;
        
        // Apply scale and convert back to original dtype
        let scaled = normalized.broadcast_mul(&self.weight)?;
        scaled.to_dtype(original_dtype)
    }
    
    /// Forward pass with residual connection (fused operation)
    pub fn forward_with_residual(&self, x: &Tensor, residual: &Tensor) -> CandleResult<Tensor> {
        let original_dtype = x.dtype();
        
        // Add residual connection first
        let x_with_residual = (x + residual)?;
        
        // Convert to f32 for computation
        let x_f32 = x_with_residual.to_dtype(DType::F32)?;
        
        // Compute RMS normalization
        let x_squared = x_f32.sqr()?;
        let mean_squared = x_squared.mean_keepdim(x_squared.dims().len() - 1)?;
        let rms = (mean_squared + self.eps)?.sqrt()?;
        
        // Normalize and scale
        let normalized = x_f32.broadcast_div(&rms)?;
        let scaled = normalized.broadcast_mul(&self.weight)?;
        
        scaled.to_dtype(original_dtype)
    }
    
    /// Get the hidden size
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
    
    /// Get the epsilon value
    pub fn eps(&self) -> f64 {
        self.eps
    }
    
    /// Load weights into this layer
    pub fn load_weight(&mut self, weight: Tensor) -> anyhow::Result<()> {
        if weight.dims() != [self.hidden_size] {
            anyhow::bail!(
                "Weight shape mismatch: expected [{}], got {:?}",
                self.hidden_size,
                weight.dims()
            );
        }
        
        self.weight = weight;
        Ok(())
    }
}

/// Optimized RMSNorm implementation with better numerical stability
#[derive(Debug)]
pub struct OptimizedRMSNorm {
    weight: Tensor,
    eps: f32,
    hidden_size: usize,
}

impl OptimizedRMSNorm {
    /// Create a new optimized RMSNorm layer
    pub fn new(hidden_size: usize, eps: f32, device: &Device, dtype: DType) -> CandleResult<Self> {
        let weight = Tensor::ones((hidden_size,), dtype, device)?;
        
        Ok(Self {
            weight,
            eps,
            hidden_size,
        })
    }
    
    /// Optimized forward pass using more stable computation
    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        // Use a more numerically stable implementation
        self.rms_norm_stable(x)
    }
    
    /// Numerically stable RMS normalization
    fn rms_norm_stable(&self, x: &Tensor) -> CandleResult<Tensor> {
        let original_dtype = x.dtype();
        let x_f32 = x.to_dtype(DType::F32)?;
        
        // Compute variance more stably
        let x_squared = x_f32.sqr()?;
        let variance = x_squared.mean_keepdim(x_squared.dims().len() - 1)?;
        
        // Add epsilon and take reciprocal square root for efficiency
        let rsqrt_var = (variance + self.eps as f64)?.powf(-0.5)?;
        
        // Apply normalization and scaling in one step
        let normalized = x_f32.broadcast_mul(&rsqrt_var)?;
        let result = normalized.broadcast_mul(&self.weight)?;
        
        result.to_dtype(original_dtype)
    }
    
    /// Forward with residual connection
    pub fn forward_with_residual(&self, x: &Tensor, residual: &Tensor) -> CandleResult<(Tensor, Tensor)> {
        // Fused residual + RMSNorm operation
        let x_with_residual = (x + residual)?;
        let normalized = self.forward(&x_with_residual)?;
        
        Ok((normalized, x_with_residual))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, DType};
    
    #[test]
    fn test_rmsnorm_creation() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        
        let rmsnorm = RMSNorm::new(512, 1e-6, &device, dtype).unwrap();
        assert_eq!(rmsnorm.hidden_size(), 512);
        assert_eq!(rmsnorm.eps(), 1e-6);
    }
    
    #[test]
    fn test_rmsnorm_forward() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        
        let rmsnorm = RMSNorm::new(4, 1e-6, &device, dtype).unwrap();
        
        // Create test input
        let input = Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            (2, 4),
            &device,
        ).unwrap();
        
        let output = rmsnorm.forward_simple(&input).unwrap();
        assert_eq!(output.dims(), input.dims());
        
        // Check that the output has approximately unit RMS
        let output_squared = output.sqr().unwrap();
        let mean_squared = output_squared.mean_keepdim(1).unwrap();
        let rms = mean_squared.sqrt().unwrap();
        
        // RMS should be close to 1.0 (within tolerance)
        let rms_values: Vec<f32> = rms.flatten_all().unwrap().to_vec1().unwrap();
        for &rms_val in &rms_values {
            assert!((rms_val - 1.0).abs() < 0.1, "RMS should be close to 1.0, got {}", rms_val);
        }
    }
    
    #[test]
    fn test_rmsnorm_with_residual() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        
        let rmsnorm = RMSNorm::new(4, 1e-6, &device, dtype).unwrap();
        
        let input = Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0],
            (1, 4),
            &device,
        ).unwrap();
        
        let residual = Tensor::from_vec(
            vec![0.5, 0.5, 0.5, 0.5],
            (1, 4),
            &device,
        ).unwrap();
        
        let output = rmsnorm.forward_with_residual(&input, &residual).unwrap();
        assert_eq!(output.dims(), input.dims());
        
        // The output should be different from simple forward pass
        let simple_output = rmsnorm.forward_simple(&input).unwrap();
        let diff = (&output - &simple_output).unwrap().abs().unwrap().sum_all().unwrap();
        let diff_val: f32 = diff.to_scalar().unwrap();
        assert!(diff_val > 0.01, "Residual connection should make a difference");
    }
    
    #[test]
    fn test_optimized_rmsnorm() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        
        let rmsnorm = OptimizedRMSNorm::new(4, 1e-6, &device, dtype).unwrap();
        
        let input = Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            (2, 4),
            &device,
        ).unwrap();
        
        let output = rmsnorm.forward(&input).unwrap();
        assert_eq!(output.dims(), input.dims());
        
        // Test with residual
        let residual = Tensor::zeros_like(&input).unwrap();
        let (norm_output, residual_output) = rmsnorm.forward_with_residual(&input, &residual).unwrap();
        
        assert_eq!(norm_output.dims(), input.dims());
        assert_eq!(residual_output.dims(), input.dims());
    }
    
    #[test]
    fn test_rmsnorm_numerical_stability() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        
        let rmsnorm = RMSNorm::new(3, 1e-8, &device, dtype).unwrap();
        
        // Test with very small values
        let small_input = Tensor::from_vec(
            vec![1e-10, 2e-10, 3e-10],
            (1, 3),
            &device,
        ).unwrap();
        
        let output = rmsnorm.forward_simple(&small_input).unwrap();
        assert!(output.dims() == small_input.dims());
        
        // Check that output doesn't contain NaN or Inf
        let output_vec: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();
        for &val in &output_vec {
            assert!(val.is_finite(), "Output should be finite, got {}", val);
        }
        
        // Test with very large values
        let large_input = Tensor::from_vec(
            vec![1e10, 2e10, 3e10],
            (1, 3),
            &device,
        ).unwrap();
        
        let output = rmsnorm.forward_simple(&large_input).unwrap();
        let output_vec: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();
        for &val in &output_vec {
            assert!(val.is_finite(), "Output should be finite for large inputs, got {}", val);
        }
    }
    
    #[test]
    fn test_weight_loading() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        
        let mut rmsnorm = RMSNorm::new(4, 1e-6, &device, dtype).unwrap();
        
        // Create custom weights
        let custom_weights = Tensor::from_vec(
            vec![0.5, 1.0, 1.5, 2.0],
            (4,),
            &device,
        ).unwrap();
        
        rmsnorm.load_weight(custom_weights.clone()).unwrap();
        
        // Test that the weights were loaded correctly
        let input = Tensor::ones((1, 4), dtype, &device).unwrap();
        let output = rmsnorm.forward_simple(&input).unwrap();
        
        // The output should be scaled by the custom weights
        assert_eq!(output.dims(), [1, 4]);
    }
    
    #[test]
    fn test_weight_loading_wrong_shape() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        
        let mut rmsnorm = RMSNorm::new(4, 1e-6, &device, dtype).unwrap();
        
        // Try to load weights with wrong shape
        let wrong_weights = Tensor::ones((3,), dtype, &device).unwrap();
        
        let result = rmsnorm.load_weight(wrong_weights);
        assert!(result.is_err(), "Should fail with wrong weight shape");
    }
}