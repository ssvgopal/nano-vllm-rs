use nano_vllm_rs::{
    LLMEngine, LLMEngineBuilder,
    config::ModelConfig,
    error::Error,
    tensor::Tensor
};
use std::path::PathBuf;
use tempfile::tempdir;
use test_utils::init_test_logging;

const TEST_MODEL: &str = "Qwen/Qwen2-0.5B-Instruct";

#[tokio::test]
async fn test_model_save_load() {
    init_test_logging();
    
    let temp_dir = tempdir().expect("Failed to create temp dir");
    let save_path = temp_dir.path().join("saved_model");
    
    // Create and save a model
    {
        let engine = LLMEngineBuilder::new()
            .model_path(TEST_MODEL)
            .device("cpu")
            .build()
            .await
            .expect("Failed to create engine");
            
        engine.save(&save_path).await.expect("Failed to save model");
    }
    
    // Load the saved model
    let engine = LLMEngineBuilder::new()
        .model_path(save_path)
        .device("cpu")
        .build()
        .await
        .expect("Failed to load saved model");
    
    // Test that the loaded model works
    let outputs = engine.generate(
        vec!["Test input".to_string()],
        Default::default()
    ).await.expect("Generation failed");
    
    assert!(!outputs[0].text.is_empty());
}

#[tokio::test]
async fn test_partial_model_loading() {
    init_test_logging();
    
    // Test loading only specific layers
    let engine = LLMEngineBuilder::new()
        .model_path(TEST_MODEL)
        .device("cpu")
        .with_config(ModelConfig {
            num_hidden_layers: 2,  // Only load first 2 layers
            ..Default::default()
        })
        .build()
        .await;
    
    assert!(engine.is_ok(), "Failed to create engine with partial layers");
}

#[tokio::test]
async fn test_corrupted_model() {
    init_test_logging();
    
    let temp_dir = tempdir().expect("Failed to create temp dir");
    let corrupt_path = temp_dir.path().join("corrupt_model");
    
    // Create a corrupted model file
    std::fs::write(&corrupt_path, "corrupt data").expect("Failed to write corrupt file");
    
    // Try to load corrupted model
    let result = LLMEngineBuilder::new()
        .model_path(corrupt_path)
        .device("cpu")
        .build()
        .await;
    
    assert!(matches!(result, Err(Error::ModelLoadError(_))));
}

#[tokio::test]
async fn test_model_with_custom_tokens() {
    init_test_logging();
    
    let mut engine = LLMEngineBuilder::new()
        .model_path(TEST_MODEL)
        .device("cpu")
        .build()
        .await
        .expect("Failed to initialize engine");
    
    // Test with custom tokens
    let custom_tokens = vec![
        "<|custom1|>".to_string(),
        "<|custom2|>".to_string(),
    ];
    
    let outputs = engine.generate(
        custom_tokens,
        SamplingParams {
            max_tokens: Some(10),
            ..Default::default()
        }
    ).await;
    
    assert!(outputs.is_ok(), "Failed to generate with custom tokens");
}
