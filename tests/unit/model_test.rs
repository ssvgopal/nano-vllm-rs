use nano_vllm_rs::{
    models::{
        transformer::Transformer,
        ModelConfig,
    },
    tensor::Tensor,
};
use test_utils::init_test_logging;

#[test]
fn test_transformer_forward() {
    init_test_logging();
    
    // Create a small test transformer
    let config = ModelConfig {
        vocab_size: 100,
        hidden_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        intermediate_size: 128,
        max_sequence_length: 64,
        ..Default::default()
    };
    
    let device = candle_core::Device::Cpu;
    let model = Transformer::new(&config, &device).expect("Failed to create transformer");
    
    // Create test input
    let input_ids = Tensor::new(&[1, 2, 3, 4], &device).unwrap();
    
    // Run forward pass
    let output = model.forward(&input_ids, None).expect("Forward pass failed");
    
    // Verify output shape
    assert_eq!(output.dims(), &[4, 100]);  // [seq_len, vocab_size]
}

#[test]
fn test_attention_mask() {
    init_test_logging();
    
    let config = ModelConfig {
        vocab_size: 100,
        hidden_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        intermediate_size: 128,
        max_sequence_length: 64,
        ..Default::default()
    };
    
    let device = candle_core::Device::Cpu;
    let model = Transformer::new(&config, &device).expect("Failed to create transformer");
    
    // Input with padding
    let input_ids = Tensor::new(&[[1, 2, 3, 0], [5, 6, 0, 0]], &device).unwrap();
    
    // Create attention mask (1 for real tokens, 0 for padding)
    let attention_mask = Tensor::new(&[[1, 1, 1, 0], [1, 1, 0, 0]], &device).unwrap();
    
    // Run forward with attention mask
    let output = model.forward(&input_ids, Some(&attention_mask)).expect("Forward pass failed");
    
    // Verify output shape
    assert_eq!(output.dims(), &[2, 4, 100]);  // [batch_size, seq_len, vocab_size]
}

#[test]
fn test_model_config_serialization() {
    use serde_json;
    
    let config = ModelConfig {
        vocab_size: 100,
        hidden_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        intermediate_size: 128,
        max_sequence_length: 64,
        ..Default::default()
    };
    
    // Serialize to JSON
    let json = serde_json::to_string(&config).expect("Serialization failed");
    
    // Deserialize back
    let deserialized: ModelConfig = serde_json::from_str(&json).expect("Deserialization failed");
    
    // Verify all fields match
    assert_eq!(config.vocab_size, deserialized.vocab_size);
    assert_eq!(config.hidden_size, deserialized.hidden_size);
    assert_eq!(config.num_hidden_layers, deserialized.num_hidden_layers);
    assert_eq!(config.num_attention_heads, deserialized.num_attention_heads);
    assert_eq!(config.intermediate_size, deserialized.intermediate_size);
    assert_eq!(config.max_sequence_length, deserialized.max_sequence_length);
}
