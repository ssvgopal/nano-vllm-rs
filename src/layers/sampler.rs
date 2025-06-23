//! Token sampling with various strategies
//! 
//! This module implements different token sampling strategies including
//! greedy sampling, temperature sampling, top-p (nucleus), and top-k sampling.

use candle_core::{Tensor, Device, DType, Result as CandleResult};
use std::collections::BinaryHeap;
use std::cmp::Ordering;

/// Token sampler with multiple sampling strategies
#[derive(Debug)]
pub struct Sampler {
    /// Device for computations
    device: Device,
}

impl Sampler {
    /// Create a new sampler
    pub fn new(device: &Device) -> Self {
        Self {
            device: device.clone(),
        }
    }
    
    /// Sample tokens from logits using the specified parameters
    pub fn forward(
        &self,
        logits: &Tensor,
        temperatures: &Tensor,
        top_p: Option<&Tensor>,
        top_k: Option<&Tensor>,
    ) -> CandleResult<Tensor> {
        let batch_size = logits.dim(0)?;
        let vocab_size = logits.dim(1)?;
        
        // Convert logits to float32 for numerical stability
        let logits_f32 = logits.to_dtype(DType::F32)?;
        
        let mut sampled_tokens = Vec::new();
        
        for i in 0..batch_size {
            let seq_logits = logits_f32.get(i)?;
            let temperature = temperatures.get(i)?.to_scalar::<f32>()?;
            
            let top_p_val = if let Some(top_p) = top_p {
                Some(top_p.get(i)?.to_scalar::<f32>()?)
            } else {
                None
            };
            
            let top_k_val = if let Some(top_k) = top_k {
                Some(top_k.get(i)?.to_scalar::<i64>()? as usize)
            } else {
                None
            };
            
            let token = self.sample_single(
                &seq_logits,
                temperature,
                top_p_val,
                top_k_val,
            )?;
            
            sampled_tokens.push(token);
        }
        
        Tensor::from_vec(sampled_tokens, (batch_size,), &self.device)
    }
    
    /// Sample a single token from logits
    fn sample_single(
        &self,
        logits: &Tensor,
        temperature: f32,
        top_p: Option<f32>,
        top_k: Option<usize>,
    ) -> CandleResult<i64> {
        // Greedy sampling if temperature is 0
        if temperature == 0.0 {
            return self.greedy_sample(logits);
        }
        
        // Apply temperature scaling
        let scaled_logits = if temperature != 1.0 {
            (logits / temperature)?
        } else {
            logits.clone()
        };
        
        // Apply top-k filtering if specified
        let filtered_logits = if let Some(k) = top_k {
            self.apply_top_k(&scaled_logits, k)?
        } else {
            scaled_logits
        };
        
        // Apply top-p filtering if specified
        let final_logits = if let Some(p) = top_p {
            self.apply_top_p(&filtered_logits, p)?
        } else {
            filtered_logits
        };
        
        // Sample from the filtered distribution
        self.multinomial_sample(&final_logits)
    }
    
    /// Greedy sampling - select the token with highest probability
    fn greedy_sample(&self, logits: &Tensor) -> CandleResult<i64> {
        let argmax = logits.argmax_keepdim(0)?;
        argmax.to_scalar::<i64>()
    }
    
    /// Apply top-k filtering to logits
    fn apply_top_k(&self, logits: &Tensor, k: usize) -> CandleResult<Tensor> {
        let vocab_size = logits.dim(0)?;
        let k = k.min(vocab_size);
        
        // Get logits as vector for processing
        let logits_vec: Vec<f32> = logits.to_vec1()?;
        
        // Find the k-th largest value
        let mut indexed_logits: Vec<(f32, usize)> = logits_vec
            .iter()
            .enumerate()
            .map(|(i, &val)| (val, i))
            .collect();
        
        // Sort by logit value in descending order
        indexed_logits.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
        
        // Get the threshold value (k-th largest)
        let threshold = if k < indexed_logits.len() {
            indexed_logits[k - 1].0
        } else {
            f32::NEG_INFINITY
        };
        
        // Create filtered logits
        let mut filtered_logits = vec![f32::NEG_INFINITY; vocab_size];
        for (logit, idx) in indexed_logits.iter().take(k) {
            if *logit >= threshold {
                filtered_logits[*idx] = *logit;
            }
        }
        
        Tensor::from_vec(filtered_logits, (vocab_size,), &self.device)
    }
    
    /// Apply top-p (nucleus) filtering to logits
    fn apply_top_p(&self, logits: &Tensor, p: f32) -> CandleResult<Tensor> {
        let vocab_size = logits.dim(0)?;
        
        // Convert to probabilities
        let probs = candle_nn::ops::softmax_last_dim(&logits.unsqueeze(0)?)?.squeeze(0)?;
        let probs_vec: Vec<f32> = probs.to_vec1()?;
        
        // Create indexed probabilities and sort by probability (descending)
        let mut indexed_probs: Vec<(f32, usize)> = probs_vec
            .iter()
            .enumerate()
            .map(|(i, &val)| (val, i))
            .collect();
        
        indexed_probs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
        
        // Find the cutoff point where cumulative probability exceeds p
        let mut cumulative_prob = 0.0;
        let mut cutoff_idx = vocab_size;
        
        for (i, (prob, _)) in indexed_probs.iter().enumerate() {
            cumulative_prob += prob;
            if cumulative_prob >= p {
                cutoff_idx = i + 1;
                break;
            }
        }
        
        // Create filtered logits
        let logits_vec: Vec<f32> = logits.to_vec1()?;
        let mut filtered_logits = vec![f32::NEG_INFINITY; vocab_size];
        
        for (prob, idx) in indexed_probs.iter().take(cutoff_idx) {
            filtered_logits[*idx] = logits_vec[*idx];
        }
        
        Tensor::from_vec(filtered_logits, (vocab_size,), &self.device)
    }
    
    /// Sample from a multinomial distribution using the Gumbel-max trick
    fn multinomial_sample(&self, logits: &Tensor) -> CandleResult<i64> {
        let vocab_size = logits.dim(0)?;
        
        // Generate Gumbel noise
        let gumbel_noise = self.sample_gumbel(vocab_size)?;
        
        // Add Gumbel noise to logits
        let gumbel_logits = (logits + gumbel_noise)?;
        
        // Return argmax
        let argmax = gumbel_logits.argmax_keepdim(0)?;
        argmax.to_scalar::<i64>()
    }
    
    /// Sample from Gumbel distribution
    fn sample_gumbel(&self, size: usize) -> CandleResult<Tensor> {
        // Generate uniform random numbers
        let uniform = Tensor::rand(0f32, 1f32, (size,), &self.device)?;
        
        // Convert to Gumbel: -log(-log(uniform))
        let eps = 1e-8f32;
        let uniform_clamped = uniform.clamp(&Tensor::new(eps, &self.device)?, &Tensor::new(1.0 - eps, &self.device)?)?;
        let log_uniform = uniform_clamped.log()?;
        let neg_log_uniform = log_uniform.neg()?;
        let gumbel = neg_log_uniform.log()?.neg()?;
        
        Ok(gumbel)
    }
    
    /// Batch sampling with different parameters per sequence
    pub fn batch_sample(
        &self,
        logits: &Tensor,
        sampling_params: &[SamplingParams],
    ) -> CandleResult<Tensor> {
        let batch_size = logits.dim(0)?;
        assert_eq!(batch_size, sampling_params.len());
        
        // Extract parameters into tensors
        let temperatures: Vec<f32> = sampling_params.iter().map(|p| p.temperature).collect();
        let temperatures_tensor = Tensor::from_vec(temperatures, (batch_size,), &self.device)?;
        
        let top_p_tensor = if sampling_params.iter().any(|p| p.top_p.is_some()) {
            let top_p_values: Vec<f32> = sampling_params
                .iter()
                .map(|p| p.top_p.unwrap_or(1.0))
                .collect();
            Some(Tensor::from_vec(top_p_values, (batch_size,), &self.device)?)
        } else {
            None
        };
        
        let top_k_tensor = if sampling_params.iter().any(|p| p.top_k.is_some()) {
            let top_k_values: Vec<i64> = sampling_params
                .iter()
                .map(|p| p.top_k.unwrap_or(0) as i64)
                .collect();
            Some(Tensor::from_vec(top_k_values, (batch_size,), &self.device)?)
        } else {
            None
        };
        
        self.forward(logits, &temperatures_tensor, top_p_tensor.as_ref(), top_k_tensor.as_ref())
    }
}

/// Sampling parameters for a single sequence
#[derive(Debug, Clone)]
pub struct SamplingParams {
    /// Temperature for sampling (0.0 = greedy, higher = more random)
    pub temperature: f32,
    
    /// Top-p (nucleus) sampling parameter
    pub top_p: Option<f32>,
    
    /// Top-k sampling parameter
    pub top_k: Option<usize>,
    
    /// Repetition penalty (not implemented in this basic version)
    pub repetition_penalty: Option<f32>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: None,
            top_k: None,
            repetition_penalty: None,
        }
    }
}

impl SamplingParams {
    /// Create new sampling parameters
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Set temperature
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }
    
    /// Set top-p
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }
    
    /// Set top-k
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = Some(top_k);
        self
    }
    
    /// Set repetition penalty
    pub fn with_repetition_penalty(mut self, penalty: f32) -> Self {
        self.repetition_penalty = Some(penalty);
        self
    }
    
    /// Check if this is greedy sampling
    pub fn is_greedy(&self) -> bool {
        self.temperature == 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, DType};
    
    #[test]
    fn test_sampler_creation() {
        let device = Device::Cpu;
        let sampler = Sampler::new(&device);
        assert_eq!(sampler.device, device);
    }
    
    #[test]
    fn test_greedy_sampling() {
        let device = Device::Cpu;
        let sampler = Sampler::new(&device);
        
        // Create logits where token 2 has highest probability
        let logits = Tensor::from_vec(vec![1.0, 2.0, 5.0, 1.5], (4,), &device).unwrap();
        let token = sampler.greedy_sample(&logits).unwrap();
        
        assert_eq!(token, 2); // Index of maximum value
    }
    
    #[test]
    fn test_temperature_scaling() {
        let device = Device::Cpu;
        let sampler = Sampler::new(&device);
        
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], (1, 3), &device).unwrap();
        let temperatures = Tensor::from_vec(vec![0.0], (1,), &device).unwrap();
        
        let tokens = sampler.forward(&logits, &temperatures, None, None).unwrap();
        let token_val: Vec<i64> = tokens.to_vec1().unwrap();
        
        assert_eq!(token_val[0], 2); // Greedy selection (highest logit)
    }
    
    #[test]
    fn test_top_k_filtering() {
        let device = Device::Cpu;
        let sampler = Sampler::new(&device);
        
        let logits = Tensor::from_vec(vec![1.0, 5.0, 2.0, 4.0, 3.0], (5,), &device).unwrap();
        let filtered = sampler.apply_top_k(&logits, 3).unwrap();
        
        let filtered_vec: Vec<f32> = filtered.to_vec1().unwrap();
        
        // Should keep top 3 values: 5.0, 4.0, 3.0 (indices 1, 3, 4)
        assert!(filtered_vec[1] == 5.0); // Highest
        assert!(filtered_vec[3] == 4.0); // Second highest
        assert!(filtered_vec[4] == 3.0); // Third highest
        assert!(filtered_vec[0] == f32::NEG_INFINITY); // Filtered out
        assert!(filtered_vec[2] == f32::NEG_INFINITY); // Filtered out
    }
    
    #[test]
    fn test_top_p_filtering() {
        let device = Device::Cpu;
        let sampler = Sampler::new(&device);
        
        // Create logits that will result in clear probability distribution
        let logits = Tensor::from_vec(vec![0.0, 10.0, 5.0, 1.0], (4,), &device).unwrap();
        let filtered = sampler.apply_top_p(&logits, 0.9).unwrap();
        
        let filtered_vec: Vec<f32> = filtered.to_vec1().unwrap();
        
        // The highest probability tokens should be kept
        assert!(filtered_vec[1] != f32::NEG_INFINITY); // Highest logit should be kept
    }
    
    #[test]
    fn test_gumbel_sampling() {
        let device = Device::Cpu;
        let sampler = Sampler::new(&device);
        
        let gumbel = sampler.sample_gumbel(5).unwrap();
        assert_eq!(gumbel.dims(), [5]);
        
        // Gumbel samples should be finite
        let gumbel_vec: Vec<f32> = gumbel.to_vec1().unwrap();
        for val in gumbel_vec {
            assert!(val.is_finite());
        }
    }
    
    #[test]
    fn test_sampling_params() {
        let params = SamplingParams::new()
            .with_temperature(0.8)
            .with_top_p(0.9)
            .with_top_k(50);
        
        assert_eq!(params.temperature, 0.8);
        assert_eq!(params.top_p, Some(0.9));
        assert_eq!(params.top_k, Some(50));
        assert!(!params.is_greedy());
        
        let greedy_params = SamplingParams::new().with_temperature(0.0);
        assert!(greedy_params.is_greedy());
    }
    
    #[test]
    fn test_batch_sampling() {
        let device = Device::Cpu;
        let sampler = Sampler::new(&device);
        
        let logits = Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            (2, 3),
            &device
        ).unwrap();
        
        let sampling_params = vec![
            SamplingParams::new().with_temperature(0.0), // Greedy
            SamplingParams::new().with_temperature(1.0), // Random
        ];
        
        let tokens = sampler.batch_sample(&logits, &sampling_params).unwrap();
        assert_eq!(tokens.dims(), [2]);
        
        let token_vals: Vec<i64> = tokens.to_vec1().unwrap();
        assert_eq!(token_vals[0], 2); // Greedy should pick highest (index 2)
    }
    
    #[test]
    fn test_multinomial_sampling_deterministic() {
        let device = Device::Cpu;
        let sampler = Sampler::new(&device);
        
        // Create logits where one token has much higher probability
        let logits = Tensor::from_vec(vec![-10.0, 10.0, -10.0], (3,), &device).unwrap();
        
        // Even with random sampling, should almost always pick the middle token
        let mut middle_count = 0;
        for _ in 0..10 {
            let token = sampler.multinomial_sample(&logits).unwrap();
            if token == 1 {
                middle_count += 1;
            }
        }
        
        // Should pick the middle token most of the time
        assert!(middle_count >= 7); // Allow some randomness
    }
}