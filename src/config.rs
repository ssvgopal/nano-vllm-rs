//! Configuration management for nano-vLLM
//! 
//! This module handles all configuration parameters for the LLM engine,
//! including model paths, memory settings, and performance tuning options.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use anyhow::{Result, Context};

/// Main configuration struct for the LLM engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Path to the model directory
    pub model_path: PathBuf,
    
    /// Maximum number of tokens that can be batched together
    pub max_num_batched_tokens: usize,
    
    /// Maximum number of sequences that can be processed simultaneously
    pub max_num_seqs: usize,
    
    /// Maximum model sequence length
    pub max_model_len: usize,
    
    /// GPU memory utilization ratio (0.0 to 1.0)
    pub gpu_memory_utilization: f32,
    
    /// Number of tensor parallel processes
    pub tensor_parallel_size: usize,
    
    /// Whether to enforce eager execution (disable optimizations)
    pub enforce_eager: bool,
    
    /// End-of-sequence token ID
    pub eos_token_id: Option<i64>,
    
    /// KV cache block size (must be multiple of 256)
    pub kvcache_block_size: usize,
    
    /// Number of KV cache blocks (auto-calculated if -1)
    pub num_kvcache_blocks: Option<usize>,
    
    /// Device to run on ("cuda", "cpu", "metal")
    pub device: String,
    
    /// Data type for model weights ("float16", "bfloat16", "float32")
    pub dtype: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            max_num_batched_tokens: 32768,
            max_num_seqs: 512,
            max_model_len: 4096,
            gpu_memory_utilization: 0.9,
            tensor_parallel_size: 1,
            enforce_eager: false,
            eos_token_id: None,
            kvcache_block_size: 256,
            num_kvcache_blocks: None,
            device: "cuda".to_string(),
            dtype: "float16".to_string(),
        }
    }
}

impl Config {
    /// Create a new config with the specified model path
    pub fn new<P: Into<PathBuf>>(model_path: P) -> Self {
        Self {
            model_path: model_path.into(),
            ..Default::default()
        }
    }
    
    /// Validate the configuration parameters
    pub fn validate(&self) -> Result<()> {
        // Check if model path exists
        if !self.model_path.exists() {
            anyhow::bail!("Model path does not exist: {:?}", self.model_path);
        }
        
        if !self.model_path.is_dir() {
            anyhow::bail!("Model path is not a directory: {:?}", self.model_path);
        }
        
        // Validate block size
        if self.kvcache_block_size % 256 != 0 {
            anyhow::bail!("KV cache block size must be a multiple of 256, got {}", self.kvcache_block_size);
        }
        
        // Validate tensor parallel size
        if !(1..=8).contains(&self.tensor_parallel_size) {
            anyhow::bail!("Tensor parallel size must be between 1 and 8, got {}", self.tensor_parallel_size);
        }
        
        // Validate GPU memory utilization
        if !(0.0..=1.0).contains(&self.gpu_memory_utilization) {
            anyhow::bail!("GPU memory utilization must be between 0.0 and 1.0, got {}", self.gpu_memory_utilization);
        }
        
        // Validate device
        if !["cuda", "cpu", "metal"].contains(&self.device.as_str()) {
            anyhow::bail!("Unsupported device: {}", self.device);
        }
        
        // Validate dtype
        if !["float16", "bfloat16", "float32"].contains(&self.dtype.as_str()) {
            anyhow::bail!("Unsupported dtype: {}", self.dtype);
        }
        
        Ok(())
    }
    
    /// Load configuration from a file
    pub fn from_file<P: Into<PathBuf>>(path: P) -> Result<Self> {
        let path = path.into();
        let content = std::fs::read_to_string(&path)
            .with_context(|| format!("Failed to read config file: {:?}", path))?;
        
        let config: Self = serde_json::from_str(&content)
            .with_context(|| format!("Failed to parse config file: {:?}", path))?;
        
        config.validate()?;
        Ok(config)
    }
    
    /// Save configuration to a file
    pub fn save_to_file<P: Into<PathBuf>>(&self, path: P) -> Result<()> {
        let path = path.into();
        let content = serde_json::to_string_pretty(self)
            .context("Failed to serialize config")?;
        
        std::fs::write(&path, content)
            .with_context(|| format!("Failed to write config file: {:?}", path))?;
        
        Ok(())
    }
    
    /// Builder pattern methods
    pub fn with_max_num_batched_tokens(mut self, tokens: usize) -> Self {
        self.max_num_batched_tokens = tokens;
        self
    }
    
    pub fn with_max_num_seqs(mut self, seqs: usize) -> Self {
        self.max_num_seqs = seqs;
        self
    }
    
    pub fn with_max_model_len(mut self, len: usize) -> Self {
        self.max_model_len = len;
        self
    }
    
    pub fn with_gpu_memory_utilization(mut self, utilization: f32) -> Self {
        self.gpu_memory_utilization = utilization;
        self
    }
    
    pub fn with_tensor_parallel_size(mut self, size: usize) -> Self {
        self.tensor_parallel_size = size;
        self
    }
    
    pub fn with_enforce_eager(mut self, eager: bool) -> Self {
        self.enforce_eager = eager;
        self
    }
    
    pub fn with_device<S: Into<String>>(mut self, device: S) -> Self {
        self.device = device.into();
        self
    }
    
    pub fn with_dtype<S: Into<String>>(mut self, dtype: S) -> Self {
        self.dtype = dtype.into();
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    
    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.max_num_batched_tokens, 32768);
        assert_eq!(config.max_num_seqs, 512);
        assert_eq!(config.tensor_parallel_size, 1);
    }
    
    #[test]
    fn test_config_validation() {
        let temp_dir = tempdir().unwrap();
        let mut config = Config::new(temp_dir.path());
        
        // Valid config should pass
        assert!(config.validate().is_ok());
        
        // Invalid block size should fail
        config.kvcache_block_size = 100;
        assert!(config.validate().is_err());
        
        // Reset and test invalid tensor parallel size
        config.kvcache_block_size = 256;
        config.tensor_parallel_size = 10;
        assert!(config.validate().is_err());
    }
    
    #[test]
    fn test_builder_pattern() {
        let temp_dir = tempdir().unwrap();
        let config = Config::new(temp_dir.path())
            .with_max_num_seqs(256)
            .with_tensor_parallel_size(2)
            .with_device("cpu");
        
        assert_eq!(config.max_num_seqs, 256);
        assert_eq!(config.tensor_parallel_size, 2);
        assert_eq!(config.device, "cpu");
    }
}