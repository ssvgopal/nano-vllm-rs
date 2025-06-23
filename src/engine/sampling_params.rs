//! Sampling parameters for text generation
//! 
//! This module defines the parameters that control how tokens are sampled
//! during text generation, including temperature, max tokens, and EOS handling.

use serde::{Deserialize, Serialize};

/// Parameters for controlling text generation sampling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingParams {
    /// Temperature for sampling (0.0 = greedy, higher = more random)
    pub temperature: f32,
    
    /// Maximum number of tokens to generate
    pub max_tokens: usize,
    
    /// Whether to ignore end-of-sequence tokens
    pub ignore_eos: bool,
    
    /// Top-p (nucleus) sampling parameter
    pub top_p: Option<f32>,
    
    /// Top-k sampling parameter
    pub top_k: Option<usize>,
    
    /// Repetition penalty
    pub repetition_penalty: Option<f32>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            max_tokens: 64,
            ignore_eos: false,
            top_p: None,
            top_k: None,
            repetition_penalty: None,
        }
    }
}

impl SamplingParams {
    /// Create new sampling parameters with default values
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Set temperature for sampling
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }
    
    /// Set maximum number of tokens to generate
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }
    
    /// Set whether to ignore EOS tokens
    pub fn with_ignore_eos(mut self, ignore_eos: bool) -> Self {
        self.ignore_eos = ignore_eos;
        self
    }
    
    /// Set top-p sampling parameter
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }
    
    /// Set top-k sampling parameter
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = Some(top_k);
        self
    }
    
    /// Set repetition penalty
    pub fn with_repetition_penalty(mut self, penalty: f32) -> Self {
        self.repetition_penalty = Some(penalty);
        self
    }
    
    /// Check if sampling is greedy (temperature == 0.0)
    pub fn is_greedy(&self) -> bool {
        self.temperature == 0.0
    }
    
    /// Validate sampling parameters
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.temperature < 0.0 {
            anyhow::bail!("Temperature must be non-negative, got {}", self.temperature);
        }
        
        if self.max_tokens == 0 {
            anyhow::bail!("Max tokens must be positive, got {}", self.max_tokens);
        }
        
        if let Some(top_p) = self.top_p {
            if !(0.0..=1.0).contains(&top_p) {
                anyhow::bail!("Top-p must be between 0.0 and 1.0, got {}", top_p);
            }
        }
        
        if let Some(top_k) = self.top_k {
            if top_k == 0 {
                anyhow::bail!("Top-k must be positive, got {}", top_k);
            }
        }
        
        if let Some(penalty) = self.repetition_penalty {
            if penalty <= 0.0 {
                anyhow::bail!("Repetition penalty must be positive, got {}", penalty);
            }
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_default_sampling_params() {
        let params = SamplingParams::default();
        assert_eq!(params.temperature, 1.0);
        assert_eq!(params.max_tokens, 64);
        assert!(!params.ignore_eos);
        assert!(params.top_p.is_none());
        assert!(params.top_k.is_none());
        assert!(params.repetition_penalty.is_none());
    }
    
    #[test]
    fn test_builder_pattern() {
        let params = SamplingParams::new()
            .with_temperature(0.8)
            .with_max_tokens(256)
            .with_top_p(0.9)
            .with_top_k(50);
        
        assert_eq!(params.temperature, 0.8);
        assert_eq!(params.max_tokens, 256);
        assert_eq!(params.top_p, Some(0.9));
        assert_eq!(params.top_k, Some(50));
    }
    
    #[test]
    fn test_validation() {
        // Valid parameters should pass
        let valid_params = SamplingParams::new().with_temperature(0.8);
        assert!(valid_params.validate().is_ok());
        
        // Invalid temperature should fail
        let invalid_temp = SamplingParams::new().with_temperature(-1.0);
        assert!(invalid_temp.validate().is_err());
        
        // Invalid max_tokens should fail
        let invalid_max_tokens = SamplingParams::new().with_max_tokens(0);
        assert!(invalid_max_tokens.validate().is_err());
        
        // Invalid top_p should fail
        let invalid_top_p = SamplingParams::new().with_top_p(1.5);
        assert!(invalid_top_p.validate().is_err());
    }
    
    #[test]
    fn test_is_greedy() {
        let greedy = SamplingParams::new().with_temperature(0.0);
        assert!(greedy.is_greedy());
        
        let non_greedy = SamplingParams::new().with_temperature(0.8);
        assert!(!non_greedy.is_greedy());
    }
}