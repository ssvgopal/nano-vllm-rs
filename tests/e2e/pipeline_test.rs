use nano_vllm_rs::{
    LLMEngine, LLMEngineBuilder, SamplingParams,
    config::ModelConfig
};
use std::time::Duration;
use tokio::time::timeout;

// This is a more comprehensive end-to-end test that verifies the complete pipeline
// from model loading to text generation with various configurations.

const TEST_TIMEOUT: Duration = Duration::from_secs(60);
const TEST_MODEL: &str = "Qwen/Qwen2-0.5B-Instruct";

#[tokio::test(flavor = "multi_thread")]
async fn test_complete_pipeline() {
    // Test with a timeout to catch hanging tests
    timeout(TEST_TIMEOUT, async {
        // 1. Initialize engine with custom config
        let mut engine = LLMEngineBuilder::new()
            .model_path(TEST_MODEL)
            .device("cpu")
            .with_config(ModelConfig {
                max_sequence_length: 512,
                max_batch_size: 4,
                use_cache: true,
                ..Default::default()
            })
            .build()
            .await
            .expect("Failed to initialize engine");

        // 2. Test basic generation
        let prompt = "The capital of France is";
        let sampling_params = SamplingParams {
            temperature: 0.7,
            top_k: 50,
            top_p: 0.9,
            max_tokens: Some(10),
            ..Default::default()
        };

        let outputs = engine.generate(
            vec![prompt.to_string()],
            sampling_params.clone()
        ).await.expect("Generation failed");

        assert!(!outputs[0].text.is_empty());
        assert!(outputs[0].tokens > 0);
        assert!(outputs[0].tokens <= 10); // Should respect max_tokens

        // 3. Test batch processing
        let prompts = vec![
            "The future of AI is".to_string(),
            "Rust programming language is".to_string(),
        ];

        let batch_outputs = engine.generate(
            prompts,
            sampling_params
        ).await.expect("Batch generation failed");

        assert_eq!(batch_outputs.len(), 2);
        assert!(!batch_outputs[0].text.is_empty());
        assert!(!batch_outputs[1].text.is_empty());

        // 4. Test streaming
        let mut stream = engine
            .generate_stream("Explain quantum computing in simple terms: ".to_string(), Default::default())
            .await
            .expect("Failed to create stream");

        let mut received_chunks = 0;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.expect("Error in stream");
            assert!(!chunk.text.is_empty());
            received_chunks += 1;
            
            if received_chunks >= 3 {
                break;
            }
        }
        
        assert!(received_chunks > 0, "No chunks received from streaming");

        // 5. Test model statistics
        let stats = engine.get_stats();
        assert!(stats.total_tokens_processed > 0);
        assert!(stats.avg_generation_time_ms > 0.0);

        // 6. Test health check
        let health = engine.check_health().await;
        assert_eq!(health.status, nano_vllm_rs::engine::HealthStatus::Healthy);
        assert!(health.memory_usage_bytes > 0);
    })
    .await
    .expect("Test timed out");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_sampling_parameters() {
    timeout(TEST_TIMEOUT, async {
        let mut engine = LLMEngineBuilder::new()
            .model_path(TEST_MODEL)
            .device("cpu")
            .build()
            .await
            .expect("Failed to initialize engine");

        let test_cases = vec![
            // Test temperature
            (SamplingParams { temperature: 0.1, ..Default::default() }, "Low temperature"),
            (SamplingParams { temperature: 1.0, ..Default::default() }, "Medium temperature"),
            (SamplingParams { temperature: 2.0, ..Default::default() }, "High temperature"),
            // Test top-k
            (SamplingParams { top_k: 10, ..Default::default() }, "Top-k=10"),
            (SamplingParams { top_k: 50, ..Default::default() }, "Top-k=50"),
            // Test top-p
            (SamplingParams { top_p: 0.5, ..Default::default() }, "Top-p=0.5"),
            (SamplingParams { top_p: 0.9, ..Default::default() }, "Top-p=0.9"),
            // Test max tokens
            (SamplingParams { max_tokens: Some(5), ..Default::default() }, "Max tokens=5"),
            (SamplingParams { max_tokens: Some(10), ..Default::default() }, "Max tokens=10"),
        ];

        for (params, description) in test_cases {
            println!("Testing: {}", description);
            let outputs = engine.generate(
                vec!["The weather today is".to_string()],
                params.clone()
            ).await.expect(&format!("Generation failed for {}", description));

            assert!(!outputs[0].text.is_empty(), "Empty output for {}", description);
            
            if let Some(max_tokens) = params.max_tokens {
                assert!(
                    outputs[0].tokens <= max_tokens as usize,
                    "Exceeded max_tokens for {}",
                    description
                );
            }
        }
    })
    .await
    .expect("Test timed out");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_error_handling() {
    timeout(TEST_TIMEOUT, async {
        // Test with invalid model path
        let engine = LLMEngineBuilder::new()
            .model_path("non/existent/path")
            .device("cpu")
            .build()
            .await;
        
        assert!(engine.is_err(), "Should fail with invalid model path");

        // Test with valid model but invalid device
        let engine = LLMEngineBuilder::new()
            .model_path(TEST_MODEL)
            .device("invalid_device")
            .build()
            .await;
            
        assert!(engine.is_err(), "Should fail with invalid device");
    })
    .await
    .expect("Test timed out");
}
