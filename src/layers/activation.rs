//! Activation functions with fused operations
//! 
//! This module implements various activation functions used in transformer models,
//! with optimizations for performance including fused operations.

use candle_core::{Tensor, Result as CandleResult};
use candle_nn::ops;

/// SiLU (Swish) activation function
/// 
/// SiLU(x) = x * sigmoid(x)
pub fn silu(x: &Tensor) -> CandleResult<Tensor> {
    let sigmoid_x = ops::sigmoid(x)?;
    x * sigmoid_x
}

/// GELU activation function (approximate)
/// 
/// GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
pub fn gelu(x: &Tensor) -> CandleResult<Tensor> {
    ops::gelu(x)
}

/// ReLU activation function
pub fn relu(x: &Tensor) -> CandleResult<Tensor> {
    ops::relu(x)
}

/// Fused SiLU and multiplication operation
/// 
/// This is commonly used in transformer MLP layers where we have:
/// gate_proj(x) * up_proj(x) where gate_proj uses SiLU activation
/// 
/// Instead of: silu(gate) * up
/// We compute: silu_and_mul([gate, up]) for efficiency
#[derive(Debug)]
pub struct SiluAndMul;

impl SiluAndMul {
    /// Create a new SiluAndMul activation
    pub fn new() -> Self {
        Self
    }
    
    /// Forward pass: split input in half, apply SiLU to first half, multiply with second half
    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let last_dim = x.dims().len() - 1;
        let dim_size = x.dim(last_dim)?;
        
        if dim_size % 2 != 0 {
            anyhow::bail!("Input dimension must be even for SiluAndMul, got {}", dim_size);
        }
        
        let half_dim = dim_size / 2;
        
        // Split the tensor in half along the last dimension
        let gate = x.narrow(last_dim, 0, half_dim)?;
        let up = x.narrow(last_dim, half_dim, half_dim)?;
        
        // Apply SiLU to gate and multiply with up
        let silu_gate = silu(&gate)?;
        silu_gate * up
    }
}

impl Default for SiluAndMul {
    fn default() -> Self {
        Self::new()
    }
}

/// Fused GELU and multiplication operation
#[derive(Debug)]
pub struct GeluAndMul;

impl GeluAndMul {
    /// Create a new GeluAndMul activation
    pub fn new() -> Self {
        Self
    }
    
    /// Forward pass: split input in half, apply GELU to first half, multiply with second half
    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let last_dim = x.dims().len() - 1;
        let dim_size = x.dim(last_dim)?;
        
        if dim_size % 2 != 0 {
            anyhow::bail!("Input dimension must be even for GeluAndMul, got {}", dim_size);
        }
        
        let half_dim = dim_size / 2;
        
        // Split the tensor in half along the last dimension
        let gate = x.narrow(last_dim, 0, half_dim)?;
        let up = x.narrow(last_dim, half_dim, half_dim)?;
        
        // Apply GELU to gate and multiply with up
        let gelu_gate = gelu(&gate)?;
        gelu_gate * up
    }
}

impl Default for GeluAndMul {
    fn default() -> Self {
        Self::new()
    }
}

/// Activation function enum for dynamic dispatch
#[derive(Debug, Clone)]
pub enum ActivationType {
    SiLU,
    GELU,
    ReLU,
    SiluAndMul,
    GeluAndMul,
}

/// Generic activation function that can handle different types
#[derive(Debug)]
pub struct Activation {
    activation_type: ActivationType,
    silu_and_mul: Option<SiluAndMul>,
    gelu_and_mul: Option<GeluAndMul>,
}

impl Activation {
    /// Create a new activation function
    pub fn new(activation_type: ActivationType) -> Self {
        let silu_and_mul = match activation_type {
            ActivationType::SiluAndMul => Some(SiluAndMul::new()),
            _ => None,
        };
        
        let gelu_and_mul = match activation_type {
            ActivationType::GeluAndMul => Some(GeluAndMul::new()),
            _ => None,
        };
        
        Self {
            activation_type,
            silu_and_mul,
            gelu_and_mul,
        }
    }
    
    /// Forward pass through the activation function
    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        match self.activation_type {
            ActivationType::SiLU => silu(x),
            ActivationType::GELU => gelu(x),
            ActivationType::ReLU => relu(x),
            ActivationType::SiluAndMul => {
                self.silu_and_mul.as_ref().unwrap().forward(x)
            }
            ActivationType::GeluAndMul => {
                self.gelu_and_mul.as_ref().unwrap().forward(x)
            }
        }
    }
    
    /// Get the activation type
    pub fn activation_type(&self) -> &ActivationType {
        &self.activation_type
    }
}

/// Parse activation function from string
impl std::str::FromStr for ActivationType {
    type Err = anyhow::Error;
    
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "silu" | "swish" => Ok(ActivationType::SiLU),
            "gelu" => Ok(ActivationType::GELU),
            "relu" => Ok(ActivationType::ReLU),
            "silu_and_mul" | "siluandmul" => Ok(ActivationType::SiluAndMul),
            "gelu_and_mul" | "geluandmul" => Ok(ActivationType::GeluAndMul),
            _ => anyhow::bail!("Unknown activation function: {}", s),
        }
    }
}

/// Optimized activation functions with compiler hints
pub mod optimized {
    use super::*;
    
    /// Optimized SiLU implementation with potential for compiler optimization
    #[inline]
    pub fn silu_optimized(x: &Tensor) -> CandleResult<Tensor> {
        // This could be replaced with a custom kernel for better performance
        super::silu(x)
    }
    
    /// Optimized fused SiLU and multiplication
    #[inline]
    pub fn silu_and_mul_optimized(x: &Tensor) -> CandleResult<Tensor> {
        let activation = SiluAndMul::new();
        activation.forward(x)
    }
    
    /// Batch-optimized activation for multiple tensors
    pub fn batch_silu(tensors: &[&Tensor]) -> CandleResult<Vec<Tensor>> {
        tensors.iter().map(|&t| silu_optimized(t)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, DType};
    
    #[test]
    fn test_silu_activation() {
        let device = Device::Cpu;
        let x = Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], (5,), &device).unwrap();
        
        let result = silu(&x).unwrap();
        let values: Vec<f32> = result.to_vec1().unwrap();
        
        // SiLU(0) should be 0
        assert!((values[2] - 0.0).abs() < 1e-6);
        
        // SiLU should be monotonically increasing
        for i in 1..values.len() {
            assert!(values[i] > values[i-1]);
        }
    }
    
    #[test]
    fn test_gelu_activation() {
        let device = Device::Cpu;
        let x = Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], (5,), &device).unwrap();
        
        let result = gelu(&x).unwrap();
        let values: Vec<f32> = result.to_vec1().unwrap();
        
        // GELU(0) should be 0
        assert!((values[2] - 0.0).abs() < 1e-6);
        
        // GELU should be monotonically increasing
        for i in 1..values.len() {
            assert!(values[i] > values[i-1]);
        }
    }
    
    #[test]
    fn test_relu_activation() {
        let device = Device::Cpu;
        let x = Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], (5,), &device).unwrap();
        
        let result = relu(&x).unwrap();
        let values: Vec<f32> = result.to_vec1().unwrap();
        
        // ReLU should clamp negative values to 0
        assert_eq!(values[0], 0.0); // -2.0 -> 0.0
        assert_eq!(values[1], 0.0); // -1.0 -> 0.0
        assert_eq!(values[2], 0.0); // 0.0 -> 0.0
        assert_eq!(values[3], 1.0); // 1.0 -> 1.0
        assert_eq!(values[4], 2.0); // 2.0 -> 2.0
    }
    
    #[test]
    fn test_silu_and_mul() {
        let device = Device::Cpu;
        let activation = SiluAndMul::new();
        
        // Create input with even dimension (6 = 3 + 3)
        let x = Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 0.5, 1.5, 2.5],
            (1, 6),
            &device
        ).unwrap();
        
        let result = activation.forward(&x).unwrap();
        assert_eq!(result.dims(), [1, 3]); // Half the input dimension
        
        // Manually compute expected result
        let gate = Tensor::from_vec(vec![1.0, 2.0, 3.0], (1, 3), &device).unwrap();
        let up = Tensor::from_vec(vec![0.5, 1.5, 2.5], (1, 3), &device).unwrap();
        let expected_gate = silu(&gate).unwrap();
        let expected = (&expected_gate * &up).unwrap();
        
        let result_values: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();
        let expected_values: Vec<f32> = expected.flatten_all().unwrap().to_vec1().unwrap();
        
        for (r, e) in result_values.iter().zip(expected_values.iter()) {
            assert!((r - e).abs() < 1e-6, "Expected {}, got {}", e, r);
        }
    }
    
    #[test]
    fn test_gelu_and_mul() {
        let device = Device::Cpu;
        let activation = GeluAndMul::new();
        
        // Create input with even dimension
        let x = Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 0.5, 1.5, 2.5],
            (1, 6),
            &device
        ).unwrap();
        
        let result = activation.forward(&x).unwrap();
        assert_eq!(result.dims(), [1, 3]);
    }
    
    #[test]
    fn test_silu_and_mul_odd_dimension() {
        let device = Device::Cpu;
        let activation = SiluAndMul::new();
        
        // Create input with odd dimension (should fail)
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], (1, 3), &device).unwrap();
        
        let result = activation.forward(&x);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_activation_enum() {
        let device = Device::Cpu;
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], (3,), &device).unwrap();
        
        // Test SiLU
        let silu_activation = Activation::new(ActivationType::SiLU);
        let silu_result = silu_activation.forward(&x).unwrap();
        let expected_silu = silu(&x).unwrap();
        
        let silu_values: Vec<f32> = silu_result.to_vec1().unwrap();
        let expected_values: Vec<f32> = expected_silu.to_vec1().unwrap();
        
        for (r, e) in silu_values.iter().zip(expected_values.iter()) {
            assert!((r - e).abs() < 1e-6);
        }
        
        // Test GELU
        let gelu_activation = Activation::new(ActivationType::GELU);
        let gelu_result = gelu_activation.forward(&x).unwrap();
        assert_eq!(gelu_result.dims(), x.dims());
        
        // Test ReLU
        let relu_activation = Activation::new(ActivationType::ReLU);
        let relu_result = relu_activation.forward(&x).unwrap();
        assert_eq!(relu_result.dims(), x.dims());
    }
    
    #[test]
    fn test_activation_from_string() {
        assert!(matches!("silu".parse::<ActivationType>().unwrap(), ActivationType::SiLU));
        assert!(matches!("swish".parse::<ActivationType>().unwrap(), ActivationType::SiLU));
        assert!(matches!("gelu".parse::<ActivationType>().unwrap(), ActivationType::GELU));
        assert!(matches!("relu".parse::<ActivationType>().unwrap(), ActivationType::ReLU));
        assert!(matches!("silu_and_mul".parse::<ActivationType>().unwrap(), ActivationType::SiluAndMul));
        
        assert!("invalid".parse::<ActivationType>().is_err());
    }
    
    #[test]
    fn test_optimized_functions() {
        let device = Device::Cpu;
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], (3,), &device).unwrap();
        
        let result = optimized::silu_optimized(&x).unwrap();
        let expected = silu(&x).unwrap();
        
        let result_values: Vec<f32> = result.to_vec1().unwrap();
        let expected_values: Vec<f32> = expected.to_vec1().unwrap();
        
        for (r, e) in result_values.iter().zip(expected_values.iter()) {
            assert!((r - e).abs() < 1e-6);
        }
    }
    
    #[test]
    fn test_batch_silu() {
        let device = Device::Cpu;
        let x1 = Tensor::from_vec(vec![1.0, 2.0], (2,), &device).unwrap();
        let x2 = Tensor::from_vec(vec![3.0, 4.0], (2,), &device).unwrap();
        let tensors = vec![&x1, &x2];
        
        let results = optimized::batch_silu(&tensors).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].dims(), [2]);
        assert_eq!(results[1].dims(), [2]);
    }
}