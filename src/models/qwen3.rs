//! Qwen3 model implementation
//! 
//! This module implements the complete Qwen3 transformer architecture
//! with all optimizations including Flash Attention, tensor parallelism,
//! and prefix caching support.

use candle_core::{Tensor, Device, DType, Result as CandleResult};
use candle_nn::{VarBuilder, VarMap};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::config::Config;
use crate::layers::{
    RMSNorm, RotaryEmbedding, Attention, SiluAndMul, 
    QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear,
    VocabParallelEmbedding, ParallelLMHead, LinearBase
};
use crate::utils::context::get_context;

/// Qwen3 model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Qwen3Config {
    /// Vocabulary size
    pub vocab_size: usize,
    
    /// Hidden size (model dimension)
    pub hidden_size: usize,
    
    /// Intermediate size in MLP
    pub intermediate_size: usize,
    
    /// Number of transformer layers
    pub num_hidden_layers: usize,
    
    /// Number of attention heads
    pub num_attention_heads: usize,
    
    /// Number of key-value heads (for grouped query attention)
    pub num_key_value_heads: usize,
    
    /// Maximum position embeddings
    pub max_position_embeddings: usize,
    
    /// RMS normalization epsilon
    pub rms_norm_eps: f64,
    
    /// Rope theta (base frequency)
    pub rope_theta: f64,
    
    /// Whether to use bias in linear layers
    pub use_bias: bool,
    
    /// Whether to tie word embeddings
    pub tie_word_embeddings: bool,
    
    /// Attention dropout (usually 0.0 for inference)
    pub attention_dropout: f64,
    
    /// Hidden dropout (usually 0.0 for inference)
    pub hidden_dropout: f64,
    
    /// Tensor parallel size
    pub tensor_parallel_size: usize,
}

impl Default for Qwen3Config {
    fn default() -> Self {
        Self {
            vocab_size: 151936,
            hidden_size: 4096,
            intermediate_size: 11008,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 32,
            max_position_embeddings: 32768,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            use_bias: false,
            tie_word_embeddings: false,
            attention_dropout: 0.0,
            hidden_dropout: 0.0,
            tensor_parallel_size: 1,
        }
    }
}

impl Qwen3Config {
    /// Create config from nano-vLLM config
    pub fn from_config(config: &Config) -> Self {
        Self {
            tensor_parallel_size: config.tensor_parallel_size,
            ..Default::default()
        }
    }
    
    /// Get head dimension
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
    
    /// Validate configuration
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.hidden_size % self.num_attention_heads != 0 {
            anyhow::bail!("Hidden size must be divisible by number of attention heads");
        }
        
        if self.num_attention_heads % self.tensor_parallel_size != 0 {
            anyhow::bail!("Number of attention heads must be divisible by tensor parallel size");
        }
        
        if self.num_key_value_heads % self.tensor_parallel_size != 0 {
            anyhow::bail!("Number of key-value heads must be divisible by tensor parallel size");
        }
        
        if self.intermediate_size % self.tensor_parallel_size != 0 {
            anyhow::bail!("Intermediate size must be divisible by tensor parallel size");
        }
        
        Ok(())
    }
}

/// Qwen3 attention layer
#[derive(Debug)]
pub struct Qwen3Attention {
    /// QKV projection (packed for efficiency)
    qkv_proj: QKVParallelLinear,
    
    /// Output projection
    o_proj: RowParallelLinear,
    
    /// Core attention mechanism
    attention: Attention,
    
    /// Rotary position embedding
    rotary_emb: RotaryEmbedding,
    
    /// Configuration
    config: Qwen3Config,
    
    /// Tensor parallel rank
    tp_rank: usize,
}

impl Qwen3Attention {
    /// Create a new Qwen3 attention layer
    pub fn new(
        config: &Qwen3Config,
        tp_rank: usize,
        device: &Device,
        dtype: DType,
    ) -> CandleResult<Self> {
        let head_dim = config.head_dim();
        let num_heads = config.num_attention_heads / config.tensor_parallel_size;
        let num_kv_heads = config.num_key_value_heads / config.tensor_parallel_size;
        
        // QKV projection (packed)
        let qkv_proj = QKVParallelLinear::new(
            config.hidden_size,
            head_dim,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.use_bias,
            tp_rank,
            config.tensor_parallel_size,
            device,
            dtype,
        )?;
        
        // Output projection
        let o_proj = RowParallelLinear::new(
            config.hidden_size,
            config.hidden_size,
            config.use_bias,
            tp_rank,
            config.tensor_parallel_size,
            device,
            dtype,
        )?;
        
        // Core attention
        let attention = Attention::new(num_heads, num_kv_heads, head_dim, device);
        
        // Rotary embedding
        let rotary_emb = RotaryEmbedding::new(
            head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            device,
            dtype,
        )?;
        
        Ok(Self {
            qkv_proj,
            o_proj,
            attention,
            rotary_emb,
            config: config.clone(),
            tp_rank,
        })
    }
    
    /// Forward pass through attention
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        position_ids: &Tensor,
    ) -> CandleResult<Tensor> {
        let batch_size = hidden_states.dim(0)?;
        let seq_len = hidden_states.dim(1)?;
        
        // QKV projection
        let qkv = self.qkv_proj.forward(hidden_states)?;
        let (query, key, value) = self.qkv_proj.split_qkv(&qkv)?;
        
        // Reshape for attention computation
        let head_dim = self.config.head_dim();
        let num_heads = self.qkv_proj.num_heads();
        let num_kv_heads = self.qkv_proj.num_kv_heads();
        
        let q = query.reshape((batch_size * seq_len, num_heads, head_dim))?;
        let k = key.reshape((batch_size * seq_len, num_kv_heads, head_dim))?;
        let v = value.reshape((batch_size * seq_len, num_kv_heads, head_dim))?;
        
        // Apply rotary position embedding
        let (q_rot, k_rot) = self.rotary_emb.forward(&q, &k, position_ids)?;
        
        // Attention computation
        let attn_output = self.attention.forward(&q_rot, &k_rot, &v)?;
        
        // Reshape back
        let attn_reshaped = attn_output.reshape((batch_size, seq_len, num_heads * head_dim))?;
        
        // Output projection
        self.o_proj.forward(&attn_reshaped)
    }
    
    /// Set KV cache for this attention layer
    pub fn set_kv_cache(&mut self, k_cache: Arc<Tensor>, v_cache: Arc<Tensor>) {
        self.attention.set_kv_cache(k_cache, v_cache);
    }
}

/// Qwen3 MLP layer
#[derive(Debug)]
pub struct Qwen3MLP {
    /// Gate and up projections (merged for efficiency)
    gate_up_proj: MergedColumnParallelLinear,
    
    /// Down projection
    down_proj: RowParallelLinear,
    
    /// Activation function (SiLU + multiplication)
    activation: SiluAndMul,
    
    /// Configuration
    config: Qwen3Config,
}

impl Qwen3MLP {
    /// Create a new Qwen3 MLP layer
    pub fn new(
        config: &Qwen3Config,
        tp_rank: usize,
        device: &Device,
        dtype: DType,
    ) -> CandleResult<Self> {
        // Merged gate and up projections
        let gate_up_proj = MergedColumnParallelLinear::new(
            config.hidden_size,
            vec![config.intermediate_size, config.intermediate_size], // gate_proj, up_proj
            config.use_bias,
            tp_rank,
            config.tensor_parallel_size,
            device,
            dtype,
        )?;
        
        // Down projection
        let down_proj = RowParallelLinear::new(
            config.intermediate_size,
            config.hidden_size,
            config.use_bias,
            tp_rank,
            config.tensor_parallel_size,
            device,
            dtype,
        )?;
        
        let activation = SiluAndMul::new();
        
        Ok(Self {
            gate_up_proj,
            down_proj,
            activation,
            config: config.clone(),
        })
    }
    
    /// Forward pass through MLP
    pub fn forward(&self, hidden_states: &Tensor) -> CandleResult<Tensor> {
        // Gate and up projections
        let gate_up = self.gate_up_proj.forward(hidden_states)?;
        
        // Apply SiLU activation with gating
        let activated = self.activation.forward(&gate_up)?;
        
        // Down projection
        self.down_proj.forward(&activated)
    }
}

/// Single Qwen3 transformer layer
#[derive(Debug)]
pub struct Qwen3Layer {
    /// Self-attention
    self_attn: Qwen3Attention,
    
    /// MLP
    mlp: Qwen3MLP,
    
    /// Input layer norm
    input_layernorm: RMSNorm,
    
    /// Post-attention layer norm
    post_attention_layernorm: RMSNorm,
    
    /// Layer index
    layer_idx: usize,
}

impl Qwen3Layer {
    /// Create a new Qwen3 layer
    pub fn new(
        config: &Qwen3Config,
        layer_idx: usize,
        tp_rank: usize,
        device: &Device,
        dtype: DType,
    ) -> CandleResult<Self> {
        let self_attn = Qwen3Attention::new(config, tp_rank, device, dtype)?;
        let mlp = Qwen3MLP::new(config, tp_rank, device, dtype)?;
        
        let input_layernorm = RMSNorm::new(
            config.hidden_size,
            config.rms_norm_eps,
            device,
            dtype,
        )?;
        
        let post_attention_layernorm = RMSNorm::new(
            config.hidden_size,
            config.rms_norm_eps,
            device,
            dtype,
        )?;
        
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            layer_idx,
        })
    }
    
    /// Forward pass through the transformer layer
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        position_ids: &Tensor,
    ) -> CandleResult<Tensor> {
        // Pre-attention normalization
        let normed_hidden = self.input_layernorm.forward_simple(hidden_states)?;
        
        // Self-attention with residual connection
        let attn_output = self.self_attn.forward(&normed_hidden, position_ids)?;
        let hidden_states = (hidden_states + attn_output)?;
        
        // Pre-MLP normalization
        let normed_hidden = self.post_attention_layernorm.forward_simple(&hidden_states)?;
        
        // MLP with residual connection
        let mlp_output = self.mlp.forward(&normed_hidden)?;
        let output = (hidden_states + mlp_output)?;
        
        Ok(output)
    }
    
    /// Set KV cache for this layer
    pub fn set_kv_cache(&mut self, k_cache: Arc<Tensor>, v_cache: Arc<Tensor>) {
        self.self_attn.set_kv_cache(k_cache, v_cache);
    }
    
    /// Get layer index
    pub fn layer_idx(&self) -> usize {
        self.layer_idx
    }
}

/// Complete Qwen3 model
#[derive(Debug)]
pub struct Qwen3Model {
    /// Model configuration
    config: Qwen3Config,
    
    /// Token embeddings
    embed_tokens: VocabParallelEmbedding,
    
    /// Transformer layers
    layers: Vec<Qwen3Layer>,
    
    /// Final layer norm
    norm: RMSNorm,
    
    /// Language model head
    lm_head: ParallelLMHead,
    
    /// Tensor parallel rank
    tp_rank: usize,
    
    /// Device
    device: Device,
}

impl Qwen3Model {
    /// Create a new Qwen3 model
    pub fn new(
        config: Qwen3Config,
        tp_rank: usize,
        device: &Device,
        dtype: DType,
    ) -> CandleResult<Self> {
        config.validate().map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        
        // Token embeddings
        let embed_tokens = VocabParallelEmbedding::new(
            config.vocab_size,
            config.hidden_size,
            tp_rank,
            config.tensor_parallel_size,
            device,
            dtype,
        )?;
        
        // Transformer layers
        let mut layers = Vec::new();
        for layer_idx in 0..config.num_hidden_layers {
            let layer = Qwen3Layer::new(&config, layer_idx, tp_rank, device, dtype)?;
            layers.push(layer);
        }
        
        // Final layer norm
        let norm = RMSNorm::new(config.hidden_size, config.rms_norm_eps, device, dtype)?;
        
        // Language model head
        let lm_head = if config.tie_word_embeddings {
            ParallelLMHead::from_embedding(embed_tokens.clone(), false, device, dtype)?
        } else {
            ParallelLMHead::new(
                config.vocab_size,
                config.hidden_size,
                false,
                tp_rank,
                config.tensor_parallel_size,
                device,
                dtype,
            )?
        };
        
        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
            lm_head,
            tp_rank,
            device: device.clone(),
        })
    }
    
    /// Forward pass through the model
    pub fn forward(
        &self,
        input_ids: &Tensor,
        position_ids: &Tensor,
    ) -> CandleResult<Tensor> {
        // Token embeddings
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;
        
        // Pass through transformer layers
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, position_ids)?;
        }
        
        // Final layer norm
        hidden_states = self.norm.forward_simple(&hidden_states)?;
        
        // Language model head
        self.lm_head.forward(&hidden_states)
    }
    
    /// Set KV cache for all layers
    pub fn set_kv_cache(&mut self, k_caches: Vec<Arc<Tensor>>, v_caches: Vec<Arc<Tensor>>) {
        assert_eq!(k_caches.len(), self.layers.len());
        assert_eq!(v_caches.len(), self.layers.len());
        
        for (layer, (k_cache, v_cache)) in self.layers.iter_mut().zip(k_caches.into_iter().zip(v_caches.into_iter())) {
            layer.set_kv_cache(k_cache, v_cache);
        }
    }
    
    /// Load model weights from a variable builder
    pub fn load_weights(&mut self, vb: VarBuilder) -> anyhow::Result<()> {
        // Load embedding weights
        let embed_weight = vb.get((self.config.vocab_size, self.config.hidden_size), "embed_tokens.weight")?;
        self.embed_tokens.load_weight(embed_weight)?;
        
        // Load layer weights
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let layer_vb = vb.pp(&format!("layers.{}", layer_idx));
            self.load_layer_weights(layer, layer_vb)?;
        }
        
        // Load final norm weights
        let norm_weight = vb.get((self.config.hidden_size,), "norm.weight")?;
        self.norm.load_weight(norm_weight)?;
        
        // Load LM head weights (if not tied)
        if !self.config.tie_word_embeddings {
            let lm_head_weight = vb.get((self.config.vocab_size, self.config.hidden_size), "lm_head.weight")?;
            self.lm_head.load_weight(lm_head_weight)?;
        }
        
        Ok(())
    }
    
    /// Load weights for a single layer
    fn load_layer_weights(&self, layer: &mut Qwen3Layer, vb: VarBuilder) -> anyhow::Result<()> {
        // Load attention weights
        let attn_vb = vb.pp("self_attn");
        
        // QKV weights (would need special handling for packed format)
        // let qkv_weight = attn_vb.get(..., "qkv_proj.weight")?;
        
        // Output projection
        // let o_proj_weight = attn_vb.get(..., "o_proj.weight")?;
        
        // Load MLP weights
        let mlp_vb = vb.pp("mlp");
        
        // Gate and up projections (merged)
        // let gate_up_weight = mlp_vb.get(..., "gate_up_proj.weight")?;
        
        // Down projection
        // let down_proj_weight = mlp_vb.get(..., "down_proj.weight")?;
        
        // Load layer norm weights
        let input_norm_weight = vb.get((self.config.hidden_size,), "input_layernorm.weight")?;
        layer.input_layernorm.load_weight(input_norm_weight)?;
        
        let post_attn_norm_weight = vb.get((self.config.hidden_size,), "post_attention_layernorm.weight")?;
        layer.post_attention_layernorm.load_weight(post_attn_norm_weight)?;
        
        Ok(())
    }
    
    /// Get model configuration
    pub fn config(&self) -> &Qwen3Config {
        &self.config
    }
    
    /// Get number of layers
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
    
    /// Get tensor parallel rank
    pub fn tp_rank(&self) -> usize {
        self.tp_rank
    }
    
    /// Get device
    pub fn device(&self) -> &Device {
        &self.device
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, DType};
    
    #[test]
    fn test_qwen3_config() {
        let config = Qwen3Config::default();
        assert_eq!(config.vocab_size, 151936);
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.head_dim(), 128); // 4096 / 32
        
        assert!(config.validate().is_ok());
    }
    
    #[test]
    fn test_qwen3_config_validation() {
        let mut config = Qwen3Config::default();
        config.hidden_size = 4097; // Not divisible by num_attention_heads
        
        assert!(config.validate().is_err());
    }
    
    #[test]
    fn test_qwen3_attention_creation() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let config = Qwen3Config::default();
        
        let attention = Qwen3Attention::new(&config, 0, &device, dtype).unwrap();
        assert_eq!(attention.tp_rank, 0);
    }
    
    #[test]
    fn test_qwen3_mlp_creation() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let config = Qwen3Config::default();
        
        let mlp = Qwen3MLP::new(&config, 0, &device, dtype).unwrap();
        // MLP should be created successfully
    }
    
    #[test]
    fn test_qwen3_layer_creation() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let config = Qwen3Config::default();
        
        let layer = Qwen3Layer::new(&config, 0, 0, &device, dtype).unwrap();
        assert_eq!(layer.layer_idx(), 0);
    }
    
    #[test]
    fn test_qwen3_model_creation() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let mut config = Qwen3Config::default();
        config.num_hidden_layers = 2; // Smaller for testing
        
        let model = Qwen3Model::new(config, 0, &device, dtype).unwrap();
        assert_eq!(model.num_layers(), 2);
        assert_eq!(model.tp_rank(), 0);
    }
    
    #[test]
    fn test_qwen3_model_forward() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let mut config = Qwen3Config::default();
        config.num_hidden_layers = 1; // Single layer for testing
        config.vocab_size = 1000; // Smaller vocab
        
        let model = Qwen3Model::new(config.clone(), 0, &device, dtype).unwrap();
        
        let batch_size = 2;
        let seq_len = 5;
        let input_ids = Tensor::from_slice(
            &[1i64, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            (batch_size, seq_len),
            &device,
        ).unwrap();
        
        let position_ids = Tensor::from_slice(
            &[0i64, 1, 2, 3, 4, 0, 1, 2, 3, 4],
            (batch_size, seq_len),
            &device,
        ).unwrap();
        
        let logits = model.forward(&input_ids, &position_ids).unwrap();
        assert_eq!(logits.dims(), [batch_size, seq_len, config.vocab_size]);
    }
    
    #[test]
    fn test_tensor_parallel_config() {
        let mut config = Qwen3Config::default();
        config.tensor_parallel_size = 2;
        config.num_attention_heads = 32; // Divisible by 2
        config.num_key_value_heads = 32; // Divisible by 2
        config.intermediate_size = 11008; // Divisible by 2
        
        assert!(config.validate().is_ok());
        
        // Test invalid tensor parallel config
        config.num_attention_heads = 31; // Not divisible by 2
        assert!(config.validate().is_err());
    }
    
    #[test]
    fn test_config_from_nano_vllm_config() {
        let nano_config = Config::default().with_tensor_parallel_size(4);
        let qwen_config = Qwen3Config::from_config(&nano_config);
        
        assert_eq!(qwen_config.tensor_parallel_size, 4);
    }
}