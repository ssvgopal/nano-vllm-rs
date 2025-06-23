//! # Nano-vLLM Rust
//! 
//! A lightweight, high-performance vLLM implementation built from scratch in Rust.
//! 
//! ## Features
//! 
//! - 🚀 **Fast offline inference** - Comparable speeds to vLLM
//! - 🦀 **Memory safe** - Built with Rust's safety guarantees
//! - ⚡ **Optimization suite** - Prefix caching, tensor parallelism, batching
//! - 📖 **Clean codebase** - Well-structured and documented
//! 
//! ## Quick Start
//! 
//! ```rust,no_run
//! use nano_vllm_rs::{LLMEngine, SamplingParams, LLMEngineBuilder};
//! 
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Create engine with builder pattern
//!     let mut engine = LLMEngineBuilder::new()
//!         .model_path("/path/to/model")
//!         .max_num_seqs(32)
//!         .device("cuda")
//!         .build()
//!         .await?;
//!     
//!     // Generate text
//!     let prompts = vec!["Hello, Nano-vLLM!".to_string()];
//!     let sampling_params = SamplingParams::new().with_temperature(0.8);
//!     let outputs = engine.generate(prompts, sampling_params).await?;
//!     
//!     println!("Generated: {}", outputs[0].text);
//!     Ok(())
//! }
//! ```
//! 
//! ## Streaming Generation
//! 
//! ```rust,no_run
//! use nano_vllm_rs::{LLMEngine, SamplingParams};
//! 
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let mut engine = LLMEngine::from_model_path("/path/to/model").await?;
//!     
//!     let prompts = vec!["Tell me a story".to_string()];
//!     let sampling_params = SamplingParams::new();
//!     let mut stream = engine.generate_stream(prompts, sampling_params).await?;
//!     
//!     while let Some(output) = stream.recv().await {
//!         print!("{}", output.text);
//!     }
//!     
//!     Ok(())
//! }
//! ```
//! 
//! ## Advanced Configuration
//! 
//! ```rust,no_run
//! use nano_vllm_rs::{Config, LLMEngine};
//! 
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let config = Config::new("/path/to/model")
//!         .with_max_num_seqs(64)
//!         .with_max_num_batched_tokens(8192)
//!         .with_gpu_memory_utilization(0.9)
//!         .with_tensor_parallel_size(2)
//!         .with_device("cuda")
//!         .with_dtype("float16");
//!     
//!     let engine = LLMEngine::new(config).await?;
//!     Ok(())
//! }
//! ```

pub mod config;
pub mod engine;
pub mod layers;
pub mod models;
pub mod utils;

// Re-export main types for convenience
pub use config::Config;
pub use engine::llm_engine::{LLMEngine, LLMEngineBuilder, EngineStats, MemoryStats, HealthStatus};
pub use engine::sampling_params::SamplingParams;
pub use engine::sequence::{Sequence, SequenceOutput, SequenceStatus};

// Error types
pub use anyhow::{Error, Result};

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Create a simple LLM engine with default settings
/// 
/// This is a convenience function for quick setup.
pub async fn create_engine<P: AsRef<std::path::Path>>(model_path: P) -> Result<LLMEngine> {
    LLMEngine::from_model_path(model_path).await
}

/// Create an LLM engine with custom configuration
pub async fn create_engine_with_config(config: Config) -> Result<LLMEngine> {
    LLMEngine::new(config).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
    
    #[tokio::test]
    async fn test_create_engine() {
        let temp_dir = tempdir().unwrap();
        let engine = create_engine(temp_dir.path()).await.unwrap();
        assert_eq!(engine.config().model_path, temp_dir.path());
    }
    
    #[tokio::test]
    async fn test_create_engine_with_config() {
        let temp_dir = tempdir().unwrap();
        let config = Config::new(temp_dir.path()).with_device("cpu");
        let engine = create_engine_with_config(config).await.unwrap();
        assert_eq!(engine.config().device, "cpu");
    }
}