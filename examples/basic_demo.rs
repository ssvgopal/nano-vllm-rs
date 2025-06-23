//! Enhanced demo for nano-vllm-rs
//! This example demonstrates various features of the nano-vllm-rs library.

use anyhow::Result;
use nano_vllm_rs::{
    LLMEngine, LLMEngineBuilder, SamplingParams, Config,
    engine::{EngineStats, HealthStatus, SequenceOutput}
};
use std::time::{Instant, Duration};
use tokio::time::sleep;
use futures_util::StreamExt;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    
    println!("🚀 Starting nano-vllm-rs demo...");
    
    // Configuration
    let model_path = "Qwen/Qwen2-0.5B-Instruct"; // Replace with your actual model path
    let device = "cuda"; // or "cpu" if no GPU available
    
    // Create engine with builder pattern
    println!("⏳ Initializing engine...");
    let start_time = Instant::now();
    
    let mut engine = LLMEngineBuilder::new()
        .model_path(model_path)
        .device(device)
        .max_num_seqs(4)
        .max_num_batched_tokens(2048)
        .build()
        .await?;
    
    println!("✅ Engine initialized in {:.2?}", start_time.elapsed());
    
    // Print engine stats
    let stats = engine.get_stats();
    println!("\n📊 Engine Stats:");
    println!("- Device: {}", stats.device);
    println!("- Model: {}", stats.model_name);
    println!("- Max Sequence Length: {}", stats.max_sequence_length);
    
    // Demo 1: Basic text generation
    demo_basic_generation(&mut engine).await?;
    
    // Demo 2: Streaming generation
    demo_streaming_generation(&mut engine).await?;
    
    // Demo 3: Advanced sampling parameters
    demo_advanced_sampling(&mut engine).await?;
    
    // Demo 4: Multiple prompts with different parameters
    demo_multiple_prompts(&mut engine).await?;
    
    // Show memory usage
    if let Some(mem_stats) = engine.get_memory_stats() {
        println!("\n💾 Memory Usage:");
        println!("- KV Cache Usage: {:.2} MB", mem_stats.kv_cache_usage_mb);
        println!("- Total Memory: {:.2} MB", mem_stats.total_memory_mb);
        println!("- Free Memory: {:.2} MB", mem_stats.free_memory_mb);
    }
    
    // Clean up
    println!("\n🧹 Cleaning up...");
    
    Ok(())
}

async fn demo_basic_generation(engine: &mut LLMEngine) -> Result<()> {
    println!("\n🚀 Demo 1: Basic Text Generation");
    println!("----------------------------------------");
    
    let prompt = "Explain what makes Rust a great programming language for AI".to_string();
    
    let sampling_params = SamplingParams::new()
        .with_temperature(0.7)
        .with_top_p(0.9)
        .with_max_tokens(256);
    
    println!("Generating response...");
    let start = Instant::now();
    let outputs = engine.generate(vec![prompt], sampling_params).await?;
    
    print_outputs(&outputs, start.elapsed());
    Ok(())
}

async fn demo_streaming_generation(engine: &mut LLMEngine) -> Result<()> {
    println!("\n🚀 Demo 2: Streaming Generation");
    println!("----------------------------------------");
    
    let prompt = "Write a short story about a robot learning to paint".to_string();
    
    let sampling_params = SamplingParams::new()
        .with_temperature(0.8)
        .with_max_tokens(100);
    
    println!("Starting streaming generation...\n");
    println!("Prompt: {}", prompt);
    println!("\nResponse (streaming):");
    
    let start = Instant::now();
    let mut stream = engine.generate_stream(vec![prompt], sampling_params).await?;
    
    while let Some(output) = stream.next().await {
        print!("{}", output.text);
        // Simulate typing effect
        sleep(Duration::from_millis(20)).await;
    }
    
    println!("\n\nStream completed in {:.2?}", start.elapsed());
    Ok(())
}

async fn demo_advanced_sampling(engine: &mut LLMEngine) -> Result<()> {
    println!("\n🚀 Demo 3: Advanced Sampling Parameters");
    println!("----------------------------------------");
    
    let prompt = "Write a creative product description for a new AI-powered coffee maker".to_string();
    
    let sampling_params = SamplingParams::new()
        .with_temperature(0.9)
        .with_top_p(0.95)
        .with_top_k(50)
        .with_frequency_penalty(0.7)
        .with_presence_penalty(0.3)
        .with_max_tokens(150);
    
    println!("Generating with creative sampling...");
    let start = Instant::now();
    let outputs = engine.generate(vec![prompt], sampling_params).await?;
    
    print_outputs(&outputs, start.elapsed());
    Ok(())
}

async fn demo_multiple_prompts(engine: &mut LLMEngine) -> Result<()> {
    println!("\n🚀 Demo 4: Multiple Prompts with Different Parameters");
    println!("----------------------------------------");
    
    let prompts = vec![
        ("Explain quantum computing in simple terms", 0.3, 100),
        ("Write a haiku about artificial intelligence", 0.8, 50),
        ("Generate a list of 5 programming jokes", 1.0, 150),
    ];
    
    for (i, (prompt, temp, max_tokens)) in prompts.into_iter().enumerate() {
        println!("\nPrompt {}: {}", i + 1, prompt);
        
        let sampling_params = SamplingParams::new()
            .with_temperature(temp)
            .with_max_tokens(max_tokens);
            
        let start = Instant::now();
        let outputs = engine.generate(vec![prompt.to_string()], sampling_params).await?;
        print_outputs(&outputs, start.elapsed());
    }
    
    Ok(())
}

fn print_outputs(outputs: &[SequenceOutput], duration: Duration) {
    for (i, output) in outputs.iter().enumerate() {
        println!("\n📝 Output {} ({} tokens, {:.2?}):", 
                i + 1, 
                output.generated_tokens, 
                output.generation_time);
        println!("----------------------------------------");
        println!("{}", output.text);
        println!("----------------------------------------");
        println!("Tokens per second: {:.2}", 
                output.generated_tokens as f64 / duration.as_secs_f64());
    }
}
}
