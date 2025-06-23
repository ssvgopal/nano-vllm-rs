//! Advanced demo for nano-vllm-rs
//! This example demonstrates advanced usage patterns of the nano-vllm-rs library.

use anyhow::Result;
use nano_vllm_rs::{
    LLMEngine, Config, SamplingParams,
    engine::EngineStats
};
use std::time::{Instant, Duration};
use tokio::time::sleep;
use futures_util::StreamExt;
use std::sync::Arc;
use tokio::sync::Mutex;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    
    println!("🚀 Starting nano-vllm-rs advanced demo...");
    
    // Initialize engine with advanced configuration
    let mut engine = initialize_engine().await?;
    
    // Run demos
    demo_chat_completion(&mut engine).await?;
    demo_parallel_generation(&mut engine).await?;
    demo_interactive_chat(&mut engine).await?;
    
    // Show final stats
    print_final_stats(&engine);
    
    Ok(())
}

async fn initialize_engine() -> Result<LLMEngine> {
    println!("⏳ Initializing engine with advanced configuration...");
    let start_time = Instant::now();
    
    let config = Config::new("Qwen/Qwen2-0.5B-Instruct")
        .with_device("cuda")  // or "cpu"
        .with_max_num_seqs(8)
        .with_max_num_batched_tokens(4096)
        .with_gpu_memory_utilization(0.9)
        .with_dtype("float16")
        .with_enable_prefix_caching(true);
    
    let engine = LLMEngine::new(config).await?;
    println!("✅ Engine initialized in {:.2?}", start_time.elapsed());
    Ok(engine)
}

fn print_final_stats(engine: &LLMEngine) {
    let stats = engine.get_stats();
    println!("\n📊 Final Engine Stats:");
    println!("- Total requests processed: {}", stats.total_processed_requests);
    println!("- Total tokens generated: {}", stats.total_generated_tokens);
    if let Some(duration) = stats.total_processing_time.checked_sub(Duration::ZERO) {
        let tps = stats.total_generated_tokens as f64 / duration.as_secs_f64();
        println!("- Average tokens per second: {:.2}", tps);
    }
}

async fn demo_chat_completion(engine: &mut LLMEngine) -> Result<()> {
    println!("\n💬 Demo 1: Chat Completion with History");
    println!("----------------------------------------");
    
    // System message
    let system_msg = "You are a helpful AI assistant. Keep your responses concise and informative.";
    
    // Chat history with system message
    let mut messages = vec![format!("<|system|>\n{system_msg}</s>")];
    
    // Example conversation
    let conversation = vec![
        ("user", "What's the capital of France?"),
        ("assistant", "The capital of France is Paris."),
        ("user", "What about Germany?"),
    ];
    
    // Add conversation to messages
    for (role, content) in conversation {
        messages.push(format!("<|{}|>\n{}</s>", role, content));
    }
    
    // Join all messages
    let prompt = messages.join("\n");
    
    // Set up sampling parameters
    let sampling_params = SamplingParams::new()
        .with_temperature(0.7)
        .with_max_tokens(100);
    
    println!("Prompt with conversation history:");
    println!("----------------------------------------");
    println!("{}", prompt);
    println!("----------------------------------------\n");
    
    println!("Generating response...");
    let start = Instant::now();
    let outputs = engine.generate(vec![prompt], sampling_params).await?;
    
    // Print the response
    if let Some(output) = outputs.first() {
        println!("\n🤖 Assistant's response ({} tokens, {:.2?}):", 
                output.generated_tokens, output.generation_time);
        println!("----------------------------------------");
        println!("{}", output.text);
        println!("----------------------------------------");
        
        let tps = output.generated_tokens as f64 / output.generation_time.as_secs_f64();
        println!("Tokens per second: {:.2}", tps);
    }
    
    Ok(())
}

async fn demo_parallel_generation(engine: &mut LLMEngine) -> Result<()> {
    println!("\n🔄 Demo 2: Parallel Generation with Different Parameters");
    println!("----------------------------------------");
    
    // Define different prompts and parameters
    let tasks = vec![
        (
            "Explain quantum computing in simple terms",
            SamplingParams::new()
                .with_temperature(0.7)
                .with_max_tokens(150),
        ),
        (
            "Write a short poem about artificial intelligence",
            SamplingParams::new()
                .with_temperature(0.9)
                .with_max_tokens(100),
        ),
        (
            "List 3 benefits of using Rust for AI applications",
            SamplingParams::new()
                .with_temperature(0.3)
                .with_max_tokens(200),
        ),
    ];
    
    // Prepare the requests
    let mut requests = Vec::new();
    for (prompt, params) in &tasks {
        requests.push(engine.generate(
            vec![prompt.to_string()], 
            params.clone()
        ));
    }
    
    println!("Starting {} parallel generations...\n", requests.len());
    let start = Instant::now();
    
    // Execute all requests concurrently
    let results = futures::future::join_all(requests).await;
    
    // Process results
    for (i, result) in results.into_iter().enumerate() {
        match result {
            Ok(outputs) => {
                if let Some(output) = outputs.first() {
                    println!("\n📝 Task {} ({} tokens, {:.2?}):", 
                            i + 1, 
                            output.generated_tokens,
                            output.generation_time);
                    println!("----------------------------------------");
                    println!("{}", output.text);
                    println!("----------------------------------------");
                    
                    let tps = output.generated_tokens as f64 / output.generation_time.as_secs_f64();
                    println!("Tokens per second: {:.2}\n", tps);
                }
            }
            Err(e) => {
                println!("❌ Error in task {}: {}", i + 1, e);
            }
        }
    }
    
    println!("\n✅ All {} tasks completed in {:.2?}", tasks.len(), start.elapsed());
    Ok(())
}
