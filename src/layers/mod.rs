//! Neural network layers for nano-vLLM
//! 
//! This module contains all the neural network layer implementations
//! used in transformer models, including attention, linear layers,
//! normalization, and activation functions.

pub mod linear;
pub mod layernorm;
pub mod rotary_embedding;
pub mod attention;
pub mod activation;
pub mod embed_head;
pub mod sampler;

// Re-export commonly used types
pub use linear::{LinearBase, ReplicatedLinear, ColumnParallelLinear, RowParallelLinear, QKVParallelLinear, MergedColumnParallelLinear};
pub use layernorm::RMSNorm;
pub use rotary_embedding::{RotaryEmbedding, apply_rotary_emb};
pub use attention::Attention;
pub use activation::SiluAndMul;
pub use embed_head::{VocabParallelEmbedding, ParallelLMHead};
pub use sampler::Sampler;