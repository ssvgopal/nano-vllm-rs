use nano_vllm_rs::{
    LLMEngine, LLMEngineBuilder, SamplingParams
};
use std::time::{Duration, Instant};
use tokio::time::sleep;
use test_utils::init_test_logging;

const TEST_MODEL: &str = "Qwen/Qwen2-0.5B-Instruct";
const WARMUP_REQUESTS: usize = 3;
const TEST_DURATION: Duration = Duration::from_secs(30);

#[tokio::test(flavor = "multi_thread")]
async fn test_concurrent_requests() {
    init_test_logging();
    
    let mut engine = LLMEngineBuilder::new()
        .model_path(TEST_MODEL)
        .device("cpu")
        .max_num_seqs(16)  // Allow more concurrent sequences
        .build()
        .await
        .expect("Failed to initialize engine");
    
    // Warm up
    for _ in 0..WARMUP_REQUESTS {
        let _ = engine.generate(
            vec!["Warm up".to_string()],
            SamplingParams {
                max_tokens: Some(5),
                ..Default::default()
            }
        ).await;
    }
    
    // Test with concurrent requests
    let start = Instant::now();
    let mut handles = vec![];
    let mut request_count = 0;
    
    // Run for TEST_DURATION
    while start.elapsed() < TEST_DURATION {
        let engine_clone = engine.clone();
        let handle = tokio::spawn(async move {
            let result = engine_clone.generate(
                vec!["Test concurrent request".to_string()],
                SamplingParams {
                    max_tokens: Some(10),
                    ..Default::default()
                }
            ).await;
            assert!(result.is_ok(), "Request failed: {:?}", result.err());
        });
        
        handles.push(handle);
        request_count += 1;
        
        // Small delay to prevent overwhelming the system
        sleep(Duration::from_millis(10)).await;
    }
    
    // Wait for all requests to complete
    for handle in handles {
        handle.await.expect("Request task panicked");
    }
    
    println!("Completed {} requests in {:?}", request_count, start.elapsed());
    let rps = request_count as f64 / start.elapsed().as_secs_f64();
    println!("Requests per second: {:.2}", rps);
    
    // Ensure we're getting reasonable throughput
    assert!(rps > 1.0, "Throughput too low: {:.2} RPS", rps);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_memory_usage() {
    init_test_logging();
    
    let mut engine = LLMEngineBuilder::new()
        .model_path(TEST_MODEL)
        .device("cpu")
        .build()
        .await
        .expect("Failed to initialize engine");
    
    // Measure memory before and after large batch
    let before_mem = memory_stats::memory_stats()
        .expect("Failed to get memory stats");
    
    // Process a large batch
    let batch_size = 8;
    let outputs = engine.generate(
        vec!["Test memory usage".to_string(); batch_size],
        SamplingParams {
            max_tokens: Some(20),
            ..Default::default()
        }
    ).await.expect("Batch generation failed");
    
    let after_mem = memory_stats::memory_stats()
        .expect("Failed to get memory stats");
    
    let mem_used_mb = (after_mem.physical_mem - before_mem.physical_mem) as f64 / 1024.0 / 1024.0;
    println!("Memory used for batch of {}: {:.2} MB", batch_size, mem_used_mb);
    
    // Verify all outputs were generated
    assert_eq!(outputs.len(), batch_size);
    for output in outputs {
        assert!(!output.text.is_empty());
    }
}
