use nano_vllm_rs::{
    LLMEngine, LLMEngineBuilder, SamplingParams, Config,
    engine::{EngineStats, HealthStatus}
};
use std::time::Duration;
use tokio::time::sleep;

// Test model configuration for integration tests
const TEST_MODEL: &str = "Qwen/Qwen2-0.5B-Instruct";

#[tokio::test]
async fn test_engine_initialization() {
    // Test CPU initialization
    let engine = LLMEngineBuilder::new()
        .model_path(TEST_MODEL)
        .device("cpu")
        .max_num_seqs(4)
        .max_num_batched_tokens(2048)
        .build()
        .await
        .expect("Failed to initialize engine");

    let stats = engine.get_stats();
    assert_eq!(stats.device, "cpu");
    assert!(stats.max_sequence_length > 0);
}

#[tokio::test]
async fn test_text_generation() {
    let mut engine = LLMEngineBuilder::new()
        .model_path(TEST_MODEL)
        .device("cpu")
        .build()
        .await
        .expect("Failed to initialize engine");

    let prompt = "The capital of France is";
    let sampling_params = SamplingParams {
        temperature: 0.7,
        top_k: 50,
        top_p: 0.9,
        ..Default::default()
    };

    let outputs = engine.generate(
        vec![prompt.to_string()],
        sampling_params
    ).await.expect("Generation failed");

    assert!(!outputs.is_empty());
    assert!(!outputs[0].text.is_empty());
    assert!(outputs[0].tokens > 0);
}

#[tokio::test]
async fn test_streaming_generation() {
    let mut engine = LLMEngineBuilder::new()
        .model_path(TEST_MODEL)
        .device("cpu")
        .build()
        .await
        .expect("Failed to initialize engine");

    let prompt = "Rust is";
    let mut stream = engine
        .generate_stream(prompt.to_string(), Default::default())
        .await
        .expect("Failed to create stream");

    let mut received_chunks = 0;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.expect("Error in stream");
        assert!(!chunk.text.is_empty());
        received_chunks += 1;
        
        if received_chunks >= 5 {
            break;
        }
    }
    
    assert!(received_chunks > 0, "No chunks received from streaming");
}

#[tokio::test]
async fn test_batch_processing() {
    let mut engine = LLMEngineBuilder::new()
        .model_path(TEST_MODEL)
        .device("cpu")
        .max_num_seqs(4)
        .build()
        .await
        .expect("Failed to initialize engine");

    let prompts = vec![
        "The future of AI is".to_string(),
        "Rust programming language is".to_string(),
    ];

    let outputs = engine
        .generate(prompts, Default::default())
        .await
        .expect("Batch generation failed");

    assert_eq!(outputs.len(), 2);
    assert!(!outputs[0].text.is_empty());
    assert!(!outputs[1].text.is_empty());
}

#[tokio::test]
async fn test_health_check() {
    let engine = LLMEngineBuilder::new()
        .model_path(TEST_MODEL)
        .device("cpu")
        .build()
        .await
        .expect("Failed to initialize engine");

    let health = engine.check_health().await;
    assert_eq!(health.status, HealthStatus::Healthy);
    assert!(health.memory_usage_bytes > 0);
    assert!(health.uptime_seconds > 0.0);
}
