use nano_vllm_rs::{
    LLMEngine, LLMEngineBuilder, SamplingParams,
    error::Error
};
use test_utils::init_test_logging;

const TEST_MODEL: &str = "Qwen/Qwen2-0.5B-Instruct";

#[tokio::test]
async fn test_empty_input() {
    init_test_logging();
    
    let mut engine = LLMEngineBuilder::new()
        .model_path(TEST_MODEL)
        .device("cpu")
        .build()
        .await
        .expect("Failed to initialize engine");
    
    // Test empty string input
    let outputs = engine.generate(
        vec!["".to_string()],
        Default::default()
    ).await;
    
    // Should either handle empty input gracefully or return an error
    assert!(matches!(
        outputs,
        Ok(_) | Err(Error::ValidationError(_))
    ));
}

#[tokio::test]
async fn test_max_sequence_length() {
    init_test_logging();
    
    let max_len = 64;
    let mut engine = LLMEngineBuilder::new()
        .model_path(TEST_MODEL)
        .device("cpu")
        .max_sequence_length(max_len)
        .build()
        .await
        .expect("Failed to initialize engine");
    
    // Create input that's exactly max_len tokens long
    let long_prompt = "test ".repeat(max_len / 5); // Rough estimation
    
    let outputs = engine.generate(
        vec![long_prompt],
        SamplingParams {
            max_tokens: Some(1),  // Just generate 1 token
            ..Default::default()
        }
    ).await.expect("Generation failed");
    
    assert!(!outputs[0].text.is_empty());
}

#[tokio::test]
async fn test_unicode_handling() {
    init_test_logging();
    
    let mut engine = LLMEngineBuilder::new()
        .model_path(TEST_MODEL)
        .device("cpu")
        .build()
        .await
        .expect("Failed to initialize engine");
    
    // Test with various Unicode characters
    let test_cases = vec![
        "Hello 世界",
        "😊 Emoji test",
        "مرحبا بالعالم",  // Arabic
        "こんにちは世界",    // Japanese
        "👨‍👩‍👧‍👦 Family emoji",
    ];
    
    for case in test_cases {
        let outputs = engine.generate(
            vec![case.to_string()],
            SamplingParams {
                max_tokens: Some(5),
                ..Default::default()
            }
        ).await.expect("Generation failed");
        
        assert!(!outputs[0].text.is_empty());
    }
}

#[tokio::test]
async fn test_invalid_sampling_params() {
    init_test_logging();
    
    let mut engine = LLMEngineBuilder::new()
        .model_path(TEST_MODEL)
        .device("cpu")
        .build()
        .await
        .expect("Failed to initialize engine");
    
    // Test invalid temperature
    let result = engine.generate(
        vec!["Test".to_string()],
        SamplingParams {
            temperature: -1.0,  // Invalid
            ..Default::default()
        }
    ).await;
    
    assert!(matches!(result, Err(Error::ValidationError(_))));
    
    // Test invalid top-k
    let result = engine.generate(
        vec!["Test".to_string()],
        SamplingParams {
            top_k: 0,  // Invalid
            ..Default::default()
        }
    ).await;
    
    assert!(matches!(result, Err(Error::ValidationError(_))));
}
