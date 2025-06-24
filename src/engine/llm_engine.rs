//! LLM Engine - High-level inference orchestration
//! 
//! This module implements the main LLM engine that orchestrates the entire
//! inference pipeline, providing a clean API for text generation.
//!
//! # Author
//! 
//! Sai Sunkara <https://github.com/ssvgopal>

use std::sync::Arc;
use tokio::sync::Mutex;
use anyhow::Result;

use crate::config::Config;
use crate::engine::{
    Scheduler, SchedulerStats, Sequence, SequenceOutput, SequenceStatus, SamplingParams
};
use crate::engine::model_runner::ModelRunner;
use crate::utils::loader::{ModelLoader, create_standard_loader};

/// Main LLM Engine for text generation
#[derive(Debug)]
pub struct LLMEngine {
    /// Model runner for executing inference
    model_runner: Arc<Mutex<ModelRunner>>,
    
    /// Scheduler for managing sequences
    scheduler: Arc<Mutex<Scheduler>>,
    
    /// Engine configuration
    config: Config,
    
    /// Whether the engine is running
    is_running: bool,
}

impl LLMEngine {
    /// Create a new LLM engine
    pub async fn new(config: Config) -> Result<Self> {
        // Validate configuration
        config.validate()?;
        
        tracing::info!("Initializing LLM Engine with config: {:?}", config);
        
        // Create model runner
        let model_runner = ModelRunner::new(&config)?;
        tracing::info!("Model runner created successfully");
        
        // Create scheduler
        let scheduler = Scheduler::new(&config);
        tracing::info!("Scheduler created successfully");
        
        let engine = Self {
            model_runner: Arc::new(Mutex::new(model_runner)),
            scheduler: Arc::new(Mutex::new(scheduler)),
            config,
            is_running: false,
        };
        
        tracing::info!("LLM Engine initialized successfully");
        Ok(engine)
    }
    
    /// Create LLM engine from model path
    pub async fn from_model_path<P: AsRef<std::path::Path>>(model_path: P) -> Result<Self> {
        let config = Config::new(model_path);
        Self::new(config).await
    }
    
    /// Generate text from prompts
    pub async fn generate(
        &mut self,
        prompts: Vec<String>,
        sampling_params: SamplingParams,
    ) -> Result<Vec<SequenceOutput>> {
        if prompts.is_empty() {
            return Ok(Vec::new());
        }
        
        tracing::info!("Starting generation for {} prompts", prompts.len());
        
        // Tokenize prompts and create sequences
        let sequences = self.create_sequences(prompts, sampling_params).await?;
        
        // Add sequences to scheduler
        {
            let mut scheduler = self.scheduler.lock().await;
            for seq in sequences {
                scheduler.add_sequence(seq);
            }
        }
        
        // Run inference loop
        let outputs = self.run_inference_loop().await?;
        
        tracing::info!("Generation completed, {} outputs produced", outputs.len());
        Ok(outputs)
    }
    
    /// Generate streaming responses
    pub async fn generate_stream(
        &mut self,
        prompts: Vec<String>,
        sampling_params: SamplingParams,
    ) -> Result<tokio::sync::mpsc::Receiver<SequenceOutput>> {
        let (tx, rx) = tokio::sync::mpsc::channel(100);
        
        // Tokenize and create sequences
        let sequences = self.create_sequences(prompts, sampling_params).await?;
        
        // Add sequences to scheduler
        {
            let mut scheduler = self.scheduler.lock().await;
            for seq in sequences {
                scheduler.add_sequence(seq);
            }
        }
        
        // Start streaming inference in background
        let engine_clone = self.clone_for_streaming().await;
        tokio::spawn(async move {
            if let Err(e) = Self::run_streaming_inference(engine_clone, tx).await {
                tracing::error!("Streaming inference failed: {}", e);
            }
        });
        
        Ok(rx)
    }
    
    /// Run the main inference loop
    async fn run_inference_loop(&mut self) -> Result<Vec<SequenceOutput>> {
        let mut outputs = Vec::new();
        
        loop {
            // Check if all sequences are finished
            {
                let scheduler = self.scheduler.lock().await;
                if scheduler.is_finished() {
                    break;
                }
            }
            
            // Execute one step
            let step_outputs = self.step().await?;
            outputs.extend(step_outputs);
            
            // Yield control to allow other tasks
            tokio::task::yield_now().await;
        }
        
        Ok(outputs)
    }
    
    /// Execute one inference step
    pub async fn step(&mut self) -> Result<Vec<SequenceOutput>> {
        let mut outputs = Vec::new();
        
        // Schedule sequences for execution
        let (sequences, is_prefill) = {
            let mut scheduler = self.scheduler.lock().await;
            scheduler.schedule()?
        };
        
        if sequences.is_empty() {
            return Ok(outputs);
        }
        
        tracing::debug!(
            "Executing step: {} sequences, prefill={}",
            sequences.len(),
            is_prefill
        );
        
        // Execute model
        let logits = {
            let mut model_runner = self.model_runner.lock().await;
            model_runner.execute_model(&sequences, is_prefill)?
        };
        
        // Sample tokens
        let token_ids = {
            let model_runner = self.model_runner.lock().await;
            model_runner.sample_tokens(&logits, &sequences)?
        };
        
        // Process results and update sequences
        {
            let mut scheduler = self.scheduler.lock().await;
            scheduler.postprocess(sequences, token_ids)?;
            
            // Collect finished sequences
            // Note: In a real implementation, you'd track finished sequences
            // and convert them to SequenceOutput
        }
        
        Ok(outputs)
    }
    
    /// Create sequences from prompts
    async fn create_sequences(
        &self,
        prompts: Vec<String>,
        sampling_params: SamplingParams,
    ) -> Result<Vec<Sequence>> {
        let mut sequences = Vec::new();
        
        for prompt in prompts {
            // Tokenize prompt (simplified - in practice use a real tokenizer)
            let token_ids = self.tokenize(&prompt).await?;
            
            // Create sequence
            let sequence = Sequence::new(token_ids, sampling_params.clone());
            sequences.push(sequence);
        }
        
        Ok(sequences)
    }
    
    /// Tokenize text (simplified implementation)
    async fn tokenize(&self, text: &str) -> Result<Vec<i64>> {
        // This is a placeholder implementation
        // In practice, you'd use a real tokenizer like HuggingFace tokenizers
        let tokens: Vec<i64> = text
            .chars()
            .map(|c| c as u32 as i64)
            .take(100) // Limit length for testing
            .collect();
        
        Ok(tokens)
    }
    
    /// Clone engine state for streaming
    async fn clone_for_streaming(&self) -> StreamingEngineState {
        StreamingEngineState {
            model_runner: Arc::clone(&self.model_runner),
            scheduler: Arc::clone(&self.scheduler),
            config: self.config.clone(),
        }
    }
    
    /// Run streaming inference
    async fn run_streaming_inference(
        state: StreamingEngineState,
        tx: tokio::sync::mpsc::Sender<SequenceOutput>,
    ) -> Result<()> {
        loop {
            // Check if finished
            {
                let scheduler = state.scheduler.lock().await;
                if scheduler.is_finished() {
                    break;
                }
            }
            
            // Execute step
            let step_outputs = Self::execute_streaming_step(&state).await?;
            
            // Send outputs
            for output in step_outputs {
                if tx.send(output).await.is_err() {
                    // Receiver dropped, stop streaming
                    return Ok(());
                }
            }
            
            tokio::task::yield_now().await;
        }
        
        Ok(())
    }
    
    /// Execute one streaming step
    async fn execute_streaming_step(state: &StreamingEngineState) -> Result<Vec<SequenceOutput>> {
        // Similar to step() but for streaming context
        let (sequences, is_prefill) = {
            let mut scheduler = state.scheduler.lock().await;
            scheduler.schedule()?
        };
        
        if sequences.is_empty() {
            return Ok(Vec::new());
        }
        
        let logits = {
            let mut model_runner = state.model_runner.lock().await;
            model_runner.execute_model(&sequences, is_prefill)?
        };
        
        let token_ids = {
            let model_runner = state.model_runner.lock().await;
            model_runner.sample_tokens(&logits, &sequences)?
        };
        
        {
            let mut scheduler = state.scheduler.lock().await;
            scheduler.postprocess(sequences, token_ids)?;
        }
        
        // Return partial outputs for streaming
        Ok(Vec::new()) // Placeholder
    }
    
    /// Load model weights
    pub async fn load_weights<P: AsRef<std::path::Path>>(&mut self, weights_path: P) -> Result<()> {
        let mut model_runner = self.model_runner.lock().await;
        model_runner.load_weights(weights_path.as_ref())?;
        tracing::info!("Model weights loaded successfully");
        Ok(())
    }
    
    /// Get engine statistics
    pub async fn get_stats(&self) -> EngineStats {
        let scheduler = self.scheduler.lock().await;
        let scheduler_stats = scheduler.get_stats();
        let block_stats = scheduler.get_block_stats();
        
        EngineStats {
            scheduler: scheduler_stats,
            memory: MemoryStats {
                total_blocks: block_stats.total_blocks,
                free_blocks: block_stats.free_blocks,
                used_blocks: block_stats.used_blocks,
                utilization: block_stats.utilization(),
            },
            is_running: self.is_running,
        }
    }
    
    /// Check if engine is healthy
    pub async fn health_check(&self) -> HealthStatus {
        let stats = self.get_stats().await;
        
        let memory_pressure = stats.memory.utilization;
        let is_healthy = memory_pressure < 95.0; // 95% threshold
        
        HealthStatus {
            is_healthy,
            memory_pressure,
            active_sequences: stats.scheduler.running_sequences,
            waiting_sequences: stats.scheduler.waiting_sequences,
        }
    }
    
    /// Shutdown the engine gracefully
    pub async fn shutdown(&mut self) -> Result<()> {
        tracing::info!("Shutting down LLM Engine");
        
        // Preempt all running sequences
        {
            let mut scheduler = self.scheduler.lock().await;
            scheduler.preempt_all();
        }
        
        self.is_running = false;
        tracing::info!("LLM Engine shutdown complete");
        Ok(())
    }
    
    /// Get configuration
    pub fn config(&self) -> &Config {
        &self.config
    }
}

/// State for streaming inference
#[derive(Debug)]
struct StreamingEngineState {
    model_runner: Arc<Mutex<ModelRunner>>,
    scheduler: Arc<Mutex<Scheduler>>,
    config: Config,
}

/// Engine statistics
#[derive(Debug, Clone)]
pub struct EngineStats {
    /// Scheduler statistics
    pub scheduler: SchedulerStats,
    
    /// Memory statistics
    pub memory: MemoryStats,
    
    /// Whether engine is running
    pub is_running: bool,
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Total number of memory blocks
    pub total_blocks: usize,
    
    /// Number of free blocks
    pub free_blocks: usize,
    
    /// Number of used blocks
    pub used_blocks: usize,
    
    /// Memory utilization percentage
    pub utilization: f64,
}

/// Health status of the engine
#[derive(Debug, Clone)]
pub struct HealthStatus {
    /// Whether the engine is healthy
    pub is_healthy: bool,
    
    /// Memory pressure (0-100%)
    pub memory_pressure: f64,
    
    /// Number of active sequences
    pub active_sequences: usize,
    
    /// Number of waiting sequences
    pub waiting_sequences: usize,
}

/// Builder for creating LLM engines with custom configurations
pub struct LLMEngineBuilder {
    config: Config,
}

impl LLMEngineBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            config: Config::default(),
        }
    }
    
    /// Set model path
    pub fn model_path<P: Into<std::path::PathBuf>>(mut self, path: P) -> Self {
        self.config.model_path = path.into();
        self
    }
    
    /// Set maximum number of sequences
    pub fn max_num_seqs(mut self, max_seqs: usize) -> Self {
        self.config.max_num_seqs = max_seqs;
        self
    }
    
    /// Set maximum number of batched tokens
    pub fn max_num_batched_tokens(mut self, max_tokens: usize) -> Self {
        self.config.max_num_batched_tokens = max_tokens;
        self
    }
    
    /// Set GPU memory utilization
    pub fn gpu_memory_utilization(mut self, utilization: f32) -> Self {
        self.config.gpu_memory_utilization = utilization;
        self
    }
    
    /// Set tensor parallel size
    pub fn tensor_parallel_size(mut self, tp_size: usize) -> Self {
        self.config.tensor_parallel_size = tp_size;
        self
    }
    
    /// Set device
    pub fn device<S: Into<String>>(mut self, device: S) -> Self {
        self.config.device = device.into();
        self
    }
    
    /// Set data type
    pub fn dtype<S: Into<String>>(mut self, dtype: S) -> Self {
        self.config.dtype = dtype.into();
        self
    }
    
    /// Enable or disable eager execution
    pub fn enforce_eager(mut self, eager: bool) -> Self {
        self.config.enforce_eager = eager;
        self
    }
    
    /// Build the LLM engine
    pub async fn build(self) -> Result<LLMEngine> {
        LLMEngine::new(self.config).await
    }
}

impl Default for LLMEngineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    
    async fn create_test_engine() -> LLMEngine {
        let temp_dir = tempdir().unwrap();
        let config = Config::new(temp_dir.path())
            .with_device("cpu")
            .with_dtype("float32")
            .with_max_num_seqs(4);
        
        LLMEngine::new(config).await.unwrap()
    }
    
    #[tokio::test]
    async fn test_engine_creation() {
        let engine = create_test_engine().await;
        assert_eq!(engine.config.device, "cpu");
        assert_eq!(engine.config.dtype, "float32");
    }
    
    #[tokio::test]
    async fn test_engine_builder() {
        let temp_dir = tempdir().unwrap();
        
        let engine = LLMEngineBuilder::new()
            .model_path(temp_dir.path())
            .max_num_seqs(8)
            .device("cpu")
            .dtype("float32")
            .build()
            .await
            .unwrap();
        
        assert_eq!(engine.config.max_num_seqs, 8);
        assert_eq!(engine.config.device, "cpu");
    }
    
    #[tokio::test]
    async fn test_tokenization() {
        let engine = create_test_engine().await;
        let tokens = engine.tokenize("hello").await.unwrap();
        
        assert_eq!(tokens.len(), 5); // "hello" has 5 characters
        assert_eq!(tokens[0], 'h' as i64);
    }
    
    #[tokio::test]
    async fn test_sequence_creation() {
        let engine = create_test_engine().await;
        let prompts = vec!["hello".to_string(), "world".to_string()];
        let sampling_params = SamplingParams::new();
        
        let sequences = engine.create_sequences(prompts, sampling_params).await.unwrap();
        assert_eq!(sequences.len(), 2);
        assert_eq!(sequences[0].num_prompt_tokens, 5); // "hello"
        assert_eq!(sequences[1].num_prompt_tokens, 5); // "world"
    }
    
    #[tokio::test]
    async fn test_engine_stats() {
        let engine = create_test_engine().await;
        let stats = engine.get_stats().await;
        
        assert_eq!(stats.scheduler.total_sequences, 0);
        assert!(!stats.is_running);
    }
    
    #[tokio::test]
    async fn test_health_check() {
        let engine = create_test_engine().await;
        let health = engine.health_check().await;
        
        assert!(health.is_healthy);
        assert_eq!(health.active_sequences, 0);
        assert_eq!(health.waiting_sequences, 0);
    }
    
    #[tokio::test]
    async fn test_engine_shutdown() {
        let mut engine = create_test_engine().await;
        engine.shutdown().await.unwrap();
        assert!(!engine.is_running);
    }
    
    #[tokio::test]
    async fn test_from_model_path() {
        let temp_dir = tempdir().unwrap();
        let engine = LLMEngine::from_model_path(temp_dir.path()).await.unwrap();
        assert_eq!(engine.config.model_path, temp_dir.path());
    }
}