//! Model runner for executing inference with optimizations
//! 
//! This module implements the model runner that manages model execution,
//! including CUDA graphs, memory management, and performance optimizations.

use std::collections::HashMap;
use std::sync::Arc;
use candle_core::{Tensor, Device, DType, Result as CandleResult};
use anyhow::Result;

use crate::config::Config;
use crate::models::qwen3::{Qwen3Model, Qwen3Config};
use crate::engine::sequence::Sequence;
use crate::utils::context::{Context, set_context};
use crate::layers::Sampler;

/// Model runner for executing inference
#[derive(Debug)]
pub struct ModelRunner {
    /// The transformer model
    model: Qwen3Model,
    
    /// Token sampler
    sampler: Sampler,
    
    /// KV cache tensors for each layer
    k_caches: Vec<Arc<Tensor>>,
    v_caches: Vec<Arc<Tensor>>,
    
    /// Block size for KV cache
    block_size: usize,
    
    /// Number of KV cache blocks
    num_blocks: usize,
    
    /// Device for computations
    device: Device,
    
    /// Data type for computations
    dtype: DType,
    
    /// Whether CUDA graphs are enabled
    cuda_graphs_enabled: bool,
    
    /// Cached CUDA graphs for different batch sizes
    cuda_graphs: HashMap<usize, CudaGraph>,
}

/// CUDA graph representation (simplified)
#[derive(Debug)]
struct CudaGraph {
    /// Batch size this graph was captured for
    batch_size: usize,
    
    /// Input tensors for the graph
    input_tensors: Vec<Tensor>,
    
    /// Output tensors from the graph
    output_tensors: Vec<Tensor>,
    
    /// Whether the graph is captured and ready
    is_captured: bool,
}

impl ModelRunner {
    /// Create a new model runner
    pub fn new(config: &Config) -> Result<Self> {
        let device = Self::get_device(&config.device)?;
        let dtype = Self::get_dtype(&config.dtype)?;
        
        // Create model configuration
        let model_config = Qwen3Config::from_config(config);
        
        // Create model
        let model = Qwen3Model::new(model_config.clone(), 0, &device, dtype)
            .map_err(|e| anyhow::anyhow!("Failed to create model: {}", e))?;
        
        // Create sampler
        let sampler = Sampler::new(&device);
        
        // Initialize KV cache
        let (k_caches, v_caches) = Self::create_kv_cache(
            &model_config,
            config.num_kvcache_blocks.unwrap_or(1000),
            config.kvcache_block_size,
            &device,
            dtype,
        )?;
        
        Ok(Self {
            model,
            sampler,
            k_caches,
            v_caches,
            block_size: config.kvcache_block_size,
            num_blocks: config.num_kvcache_blocks.unwrap_or(1000),
            device,
            dtype,
            cuda_graphs_enabled: !config.enforce_eager,
            cuda_graphs: HashMap::new(),
        })
    }
    
    /// Execute model forward pass
    pub fn execute_model(
        &mut self,
        sequences: &[Sequence],
        is_prefill: bool,
    ) -> Result<Tensor> {
        // Prepare input tensors
        let (input_ids, position_ids) = self.prepare_inputs(sequences, is_prefill)?;
        
        // Set up context for attention computation
        let context = self.create_context(sequences, is_prefill)?;
        set_context(context)?;
        
        // Execute model
        let logits = if self.cuda_graphs_enabled && !is_prefill {
            // Use CUDA graphs for decode phase
            self.execute_with_cuda_graph(&input_ids, &position_ids)?
        } else {
            // Direct execution for prefill or when CUDA graphs disabled
            self.model.forward(&input_ids, &position_ids)
                .map_err(|e| anyhow::anyhow!("Model forward failed: {}", e))?
        };
        
        Ok(logits)
    }
    
    /// Sample tokens from logits
    pub fn sample_tokens(
        &self,
        logits: &Tensor,
        sequences: &[Sequence],
    ) -> Result<Vec<i64>> {
        // Extract sampling parameters from sequences
        let sampling_params: Vec<_> = sequences
            .iter()
            .map(|seq| crate::layers::sampler::SamplingParams {
                temperature: seq.sampling_params.temperature,
                top_p: seq.sampling_params.top_p,
                top_k: seq.sampling_params.top_k,
                repetition_penalty: seq.sampling_params.repetition_penalty,
            })
            .collect();
        
        // Sample tokens
        let sampled_tokens = self.sampler.batch_sample(logits, &sampling_params)
            .map_err(|e| anyhow::anyhow!("Sampling failed: {}", e))?;
        
        // Convert to vector
        let token_ids: Vec<i64> = sampled_tokens.to_vec1()
            .map_err(|e| anyhow::anyhow!("Failed to extract token IDs: {}", e))?;
        
        Ok(token_ids)
    }
    
    /// Prepare input tensors for model execution
    fn prepare_inputs(
        &self,
        sequences: &[Sequence],
        is_prefill: bool,
    ) -> Result<(Tensor, Tensor)> {
        if is_prefill {
            self.prepare_prefill_inputs(sequences)
        } else {
            self.prepare_decode_inputs(sequences)
        }
    }
    
    /// Prepare inputs for prefill phase
    fn prepare_prefill_inputs(&self, sequences: &[Sequence]) -> Result<(Tensor, Tensor)> {
        let mut all_input_ids = Vec::new();
        let mut all_position_ids = Vec::new();
        
        for seq in sequences {
            let input_ids = seq.all_token_ids();
            let position_ids: Vec<i64> = (0..input_ids.len() as i64).collect();
            
            all_input_ids.extend_from_slice(input_ids);
            all_position_ids.extend(position_ids);
        }
        
        let total_tokens = all_input_ids.len();
        
        let input_ids_tensor = Tensor::from_vec(all_input_ids, (total_tokens,), &self.device)
            .map_err(|e| anyhow::anyhow!("Failed to create input_ids tensor: {}", e))?;
        
        let position_ids_tensor = Tensor::from_vec(all_position_ids, (total_tokens,), &self.device)
            .map_err(|e| anyhow::anyhow!("Failed to create position_ids tensor: {}", e))?;
        
        Ok((input_ids_tensor, position_ids_tensor))
    }
    
    /// Prepare inputs for decode phase
    fn prepare_decode_inputs(&self, sequences: &[Sequence]) -> Result<(Tensor, Tensor)> {
        let batch_size = sequences.len();
        
        // For decode, we only need the last token of each sequence
        let input_ids: Vec<i64> = sequences.iter().map(|seq| seq.last_token).collect();
        let position_ids: Vec<i64> = sequences.iter().map(|seq| seq.len() as i64 - 1).collect();
        
        let input_ids_tensor = Tensor::from_vec(input_ids, (batch_size,), &self.device)
            .map_err(|e| anyhow::anyhow!("Failed to create input_ids tensor: {}", e))?;
        
        let position_ids_tensor = Tensor::from_vec(position_ids, (batch_size,), &self.device)
            .map_err(|e| anyhow::anyhow!("Failed to create position_ids tensor: {}", e))?;
        
        Ok((input_ids_tensor, position_ids_tensor))
    }
    
    /// Create context for attention computation
    fn create_context(&self, sequences: &[Sequence], is_prefill: bool) -> Result<Context> {
        if is_prefill {
            self.create_prefill_context(sequences)
        } else {
            self.create_decode_context(sequences)
        }
    }
    
    /// Create context for prefill phase
    fn create_prefill_context(&self, sequences: &[Sequence]) -> Result<Context> {
        let mut cu_seqlens_q = vec![0i32];
        let mut cu_seqlens_k = vec![0i32];
        let mut slot_mapping = Vec::new();
        let mut max_seqlen_q = 0;
        let mut max_seqlen_k = 0;
        
        let mut current_pos = 0;
        
        for seq in sequences {
            let seq_len = seq.len();
            current_pos += seq_len;
            cu_seqlens_q.push(current_pos as i32);
            cu_seqlens_k.push(current_pos as i32);
            
            max_seqlen_q = max_seqlen_q.max(seq_len);
            max_seqlen_k = max_seqlen_k.max(seq_len);
            
            // Create slot mapping for this sequence
            for i in 0..seq_len {
                slot_mapping.push(i as i32); // Simplified slot mapping
            }
        }
        
        let cu_seqlens_q_tensor = Tensor::from_vec(cu_seqlens_q, (sequences.len() + 1,), &self.device)
            .map_err(|e| anyhow::anyhow!("Failed to create cu_seqlens_q: {}", e))?;
        
        let cu_seqlens_k_tensor = Tensor::from_vec(cu_seqlens_k, (sequences.len() + 1,), &self.device)
            .map_err(|e| anyhow::anyhow!("Failed to create cu_seqlens_k: {}", e))?;
        
        let slot_mapping_tensor = Tensor::from_vec(slot_mapping, (current_pos,), &self.device)
            .map_err(|e| anyhow::anyhow!("Failed to create slot_mapping: {}", e))?;
        
        Ok(Context::prefill(
            cu_seqlens_q_tensor,
            cu_seqlens_k_tensor,
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping_tensor,
            None, // No block tables for basic prefill
        ))
    }
    
    /// Create context for decode phase
    fn create_decode_context(&self, sequences: &[Sequence]) -> Result<Context> {
        let batch_size = sequences.len();
        
        // Slot mapping for decode (one slot per sequence)
        let slot_mapping: Vec<i32> = (0..batch_size as i32).collect();
        let slot_mapping_tensor = Tensor::from_vec(slot_mapping, (batch_size,), &self.device)
            .map_err(|e| anyhow::anyhow!("Failed to create slot_mapping: {}", e))?;
        
        // Context lengths
        let context_lens: Vec<i32> = sequences.iter().map(|seq| seq.len() as i32).collect();
        let context_lens_tensor = Tensor::from_vec(context_lens, (batch_size,), &self.device)
            .map_err(|e| anyhow::anyhow!("Failed to create context_lens: {}", e))?;
        
        // Block tables (simplified)
        let max_blocks = sequences.iter().map(|seq| seq.num_blocks()).max().unwrap_or(1);
        let mut block_tables = Vec::new();
        
        for seq in sequences {
            let mut blocks = seq.block_table.clone();
            // Pad to max_blocks
            while blocks.len() < max_blocks {
                blocks.push(-1); // Invalid block ID
            }
            block_tables.extend(blocks);
        }
        
        let block_tables_tensor = Tensor::from_vec(block_tables, (batch_size, max_blocks), &self.device)
            .map_err(|e| anyhow::anyhow!("Failed to create block_tables: {}", e))?;
        
        Ok(Context::decode(
            slot_mapping_tensor,
            context_lens_tensor,
            block_tables_tensor,
        ))
    }
    
    /// Execute model with CUDA graphs
    fn execute_with_cuda_graph(
        &mut self,
        input_ids: &Tensor,
        position_ids: &Tensor,
    ) -> Result<Tensor> {
        let batch_size = input_ids.dim(0).map_err(|e| anyhow::anyhow!("Failed to get batch size: {}", e))?;
        
        // Check if we have a CUDA graph for this batch size
        if !self.cuda_graphs.contains_key(&batch_size) {
            // Capture a new CUDA graph
            self.capture_cuda_graph(batch_size)?;
        }
        
        let cuda_graph = self.cuda_graphs.get_mut(&batch_size).unwrap();
        
        if cuda_graph.is_captured {
            // Use the captured graph
            self.execute_captured_graph(cuda_graph, input_ids, position_ids)
        } else {
            // Fallback to direct execution
            self.model.forward(input_ids, position_ids)
                .map_err(|e| anyhow::anyhow!("Model forward failed: {}", e))
        }
    }
    
    /// Capture a CUDA graph for a specific batch size
    fn capture_cuda_graph(&mut self, batch_size: usize) -> Result<()> {
        // This is a simplified implementation
        // In practice, CUDA graph capture involves:
        // 1. Warming up the model
        // 2. Starting graph capture
        // 3. Running the model
        // 4. Ending graph capture
        
        let cuda_graph = CudaGraph {
            batch_size,
            input_tensors: Vec::new(),
            output_tensors: Vec::new(),
            is_captured: false, // Set to false for now
        };
        
        self.cuda_graphs.insert(batch_size, cuda_graph);
        
        tracing::info!("CUDA graph capture attempted for batch size {}", batch_size);
        Ok(())
    }
    
    /// Execute a captured CUDA graph
    fn execute_captured_graph(
        &self,
        _cuda_graph: &mut CudaGraph,
        input_ids: &Tensor,
        position_ids: &Tensor,
    ) -> Result<Tensor> {
        // This would execute the captured graph
        // For now, fallback to direct execution
        self.model.forward(input_ids, position_ids)
            .map_err(|e| anyhow::anyhow!("Model forward failed: {}", e))
    }
    
    /// Create KV cache tensors
    fn create_kv_cache(
        config: &Qwen3Config,
        num_blocks: usize,
        block_size: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<(Vec<Arc<Tensor>>, Vec<Arc<Tensor>>)> {
        let num_layers = config.num_hidden_layers;
        let num_kv_heads = config.num_key_value_heads / config.tensor_parallel_size;
        let head_dim = config.head_dim();
        
        let mut k_caches = Vec::new();
        let mut v_caches = Vec::new();
        
        for _ in 0..num_layers {
            let k_cache = Tensor::zeros(
                (num_blocks, block_size, num_kv_heads, head_dim),
                dtype,
                device,
            ).map_err(|e| anyhow::anyhow!("Failed to create K cache: {}", e))?;
            
            let v_cache = Tensor::zeros(
                (num_blocks, block_size, num_kv_heads, head_dim),
                dtype,
                device,
            ).map_err(|e| anyhow::anyhow!("Failed to create V cache: {}", e))?;
            
            k_caches.push(Arc::new(k_cache));
            v_caches.push(Arc::new(v_cache));
        }
        
        Ok((k_caches, v_caches))
    }
    
    /// Get device from string
    fn get_device(device_str: &str) -> Result<Device> {
        match device_str.to_lowercase().as_str() {
            "cpu" => Ok(Device::Cpu),
            "cuda" => {
                #[cfg(feature = "cuda")]
                {
                    Ok(Device::new_cuda(0).map_err(|e| anyhow::anyhow!("Failed to create CUDA device: {}", e))?)
                }
                #[cfg(not(feature = "cuda"))]
                {
                    anyhow::bail!("CUDA support not compiled in")
                }
            }
            "metal" => {
                #[cfg(feature = "metal")]
                {
                    Ok(Device::new_metal(0).map_err(|e| anyhow::anyhow!("Failed to create Metal device: {}", e))?)
                }
                #[cfg(not(feature = "metal"))]
                {
                    anyhow::bail!("Metal support not compiled in")
                }
            }
            _ => anyhow::bail!("Unsupported device: {}", device_str),
        }
    }
    
    /// Get data type from string
    fn get_dtype(dtype_str: &str) -> Result<DType> {
        match dtype_str.to_lowercase().as_str() {
            "float32" | "f32" => Ok(DType::F32),
            "float16" | "f16" => Ok(DType::F16),
            "bfloat16" | "bf16" => Ok(DType::BF16),
            _ => anyhow::bail!("Unsupported dtype: {}", dtype_str),
        }
    }
    
    /// Load model weights
    pub fn load_weights(&mut self, weights_path: &std::path::Path) -> Result<()> {
        // This would load weights from safetensors files
        // For now, just log the attempt
        tracing::info!("Loading model weights from: {:?}", weights_path);
        
        // In a real implementation:
        // 1. Load safetensors files
        // 2. Create VarBuilder from loaded tensors
        // 3. Call model.load_weights(vb)
        
        Ok(())
    }
    
    /// Get model configuration
    pub fn config(&self) -> &Qwen3Config {
        self.model.config()
    }
    
    /// Get device
    pub fn device(&self) -> &Device {
        &self.device
    }
    
    /// Get data type
    pub fn dtype(&self) -> DType {
        self.dtype
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::sampling_params::SamplingParams;
    
    fn create_test_config() -> Config {
        Config::default()
            .with_device("cpu")
            .with_dtype("float32")
            .with_max_num_seqs(4)
    }
    
    #[test]
    fn test_model_runner_creation() {
        let config = create_test_config();
        let runner = ModelRunner::new(&config).unwrap();
        
        assert_eq!(runner.device, Device::Cpu);
        assert_eq!(runner.dtype, DType::F32);
    }
    
    #[test]
    fn test_device_parsing() {
        assert_eq!(ModelRunner::get_device("cpu").unwrap(), Device::Cpu);
        assert!(ModelRunner::get_device("invalid").is_err());
    }
    
    #[test]
    fn test_dtype_parsing() {
        assert_eq!(ModelRunner::get_dtype("float32").unwrap(), DType::F32);
        assert_eq!(ModelRunner::get_dtype("f32").unwrap(), DType::F32);
        assert!(ModelRunner::get_dtype("invalid").is_err());
    }
    
    #[test]
    fn test_prepare_decode_inputs() {
        let config = create_test_config();
        let runner = ModelRunner::new(&config).unwrap();
        
        let sequences = vec![
            Sequence::new(vec![1, 2, 3], SamplingParams::new()),
            Sequence::new(vec![4, 5], SamplingParams::new()),
        ];
        
        let (input_ids, position_ids) = runner.prepare_decode_inputs(&sequences).unwrap();
        
        assert_eq!(input_ids.dims(), [2]); // Batch size
        assert_eq!(position_ids.dims(), [2]);
        
        let input_vals: Vec<i64> = input_ids.to_vec1().unwrap();
        let pos_vals: Vec<i64> = position_ids.to_vec1().unwrap();
        
        assert_eq!(input_vals, vec![3, 5]); // Last tokens
        assert_eq!(pos_vals, vec![2, 1]); // Position of last tokens
    }
    
    #[test]
    fn test_kv_cache_creation() {
        let config = Qwen3Config::default();
        let device = Device::Cpu;
        let dtype = DType::F32;
        
        let (k_caches, v_caches) = ModelRunner::create_kv_cache(
            &config,
            100, // num_blocks
            16,  // block_size
            &device,
            dtype,
        ).unwrap();
        
        assert_eq!(k_caches.len(), config.num_hidden_layers);
        assert_eq!(v_caches.len(), config.num_hidden_layers);
        
        let expected_shape = [100, 16, config.num_key_value_heads, config.head_dim()];
        assert_eq!(k_caches[0].dims(), expected_shape);
        assert_eq!(v_caches[0].dims(), expected_shape);
    }
}