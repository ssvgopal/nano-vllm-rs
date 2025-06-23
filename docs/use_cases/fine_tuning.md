# Fine-tuning Language Models with Local Data

This guide explains how to use `nano-vllm-rs` to fine-tune language models using your own JSON-formatted datasets.

## 📋 Prerequisites

- Rust toolchain (latest stable version)
- CUDA (for GPU acceleration, optional but recommended)
- Sufficient disk space for models and datasets
- At least 16GB RAM (32GB+ recommended for larger models)

## 📁 Data Preparation

### Supported JSON Formats

1. **Instruction-Following Format**
   ```json
   [
     {
       "instruction": "Write a professional email",
       "input": "Subject: Meeting Request",
       "output": "Dear Team,..."
     }
   ]
   ```

2. **Conversation Format**
   ```json
   [
     {
       "messages": [
         {"role": "system", "content": "You are a helpful assistant."},
         {"role": "user", "content": "How do I reset my password?"},
         {"role": "assistant", "content": "To reset your password..."}
       ]
     }
   ]
   ```

3. **Text Completion Format**
   ```json
   [
     {
       "text": "The quick brown fox jumps over the lazy dog."
     }
   ]
   ```

## 🛠️ Implementation

### 1. Project Setup

```bash
# Create a new Rust binary
cargo new fine_tuning_example
cd fine_tuning_example
```

### 2. Add Dependencies

```toml
# Cargo.toml
[dependencies]
nano-vllm-rs = { git = "https://github.com/your-username/nano-vllm-rs" }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
tokio = { version = "1.0", features = ["full"] }
anyhow = "1.0"
```

### 3. Data Loading Module

Create `src/data_loader.rs`:

```rust
use serde::Deserialize;
use std::fs::File;
use std::io::BufReader;
use anyhow::Result;

#[derive(Debug, Deserialize, Clone)]
pub struct TrainingExample {
    pub prompt: String,
    pub completion: String,
}

pub fn load_training_data(file_path: &str) -> Result<Vec<TrainingExample>> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    let examples: Vec<TrainingExample> = serde_json::from_reader(reader)?;
    Ok(examples)
}
```

### 4. Main Training Script

Create `src/main.rs`:

```rust
mod data_loader;

use nano_vllm_rs::{
    LLMEngine, Config, TrainingConfig,
    engine::TrainingProgress
};
use anyhow::Result;
use std::path::Path;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    env_logger::init();
    
    // 1. Load training data
    println!("Loading training data...");
    let train_examples = data_loader::load_training_data("data/train.json")?;
    let val_examples = data_loader::load_training_data("data/val.json")?;
    
    println!("Loaded {} training examples, {} validation examples", 
             train_examples.len(), val_examples.len());
    
    // 2. Initialize model
    let config = Config::new("Qwen/Qwen2-0.5B-Instruct")
        .with_device("cuda")
        .with_training(true);
    
    let mut engine = LLMEngine::new(config).await?;
    
    // 3. Configure training
    let training_config = TrainingConfig {
        learning_rate: 5e-5,
        batch_size: 4,
        num_epochs: 3,
        warmup_steps: 100,
        weight_decay: 0.01,
        output_dir: "output/finetuned_model",
        save_steps: 500,
        logging_steps: 10,
    };
    
    // 4. Start fine-tuning
    println!("Starting fine-tuning...");
    engine.fine_tune(&train_examples, Some(&val_examples), &training_config, |progress| {
        match progress {
            TrainingProgress::Step { step, loss, learning_rate } => {
                println!("Step {}: loss={:.4}, lr={:.2e}", step, loss, learning_rate);
            }
            TrainingProgress::Epoch { epoch, train_loss, val_loss } => {
                println!("\nEpoch {} complete - train_loss: {:.4}, val_loss: {:.4}\n", 
                        epoch, train_loss, val_loss.unwrap_or_default());
            }
        }
    }).await?;
    
    // 5. Save the fine-tuned model
    println!("Saving fine-tuned model...");
    engine.save_model("output/finetuned_model")?;
    
    println!("Fine-tuning complete!");
    Ok(())
}
```

## 🚀 Running the Training

1. Prepare your data:
   ```
   mkdir -p data
   # Add your train.json and val.json files here
   ```

2. Run the training:
   ```bash
   cargo run --release
   ```

## 🧪 Evaluation

Create `src/evaluate.rs` to evaluate your fine-tuned model:

```rust
use nano_vllm_rs::{LLMEngine, Config};
use anyhow::Result;

pub async fn evaluate_model(model_path: &str, test_examples: &[String]) -> Result<f64> {
    let config = Config::new(model_path)
        .with_device("cuda");
    
    let mut engine = LLMEngine::new(config).await?;
    let mut correct = 0;
    
    for (i, example) in test_examples.iter().enumerate() {
        // Implement your evaluation logic here
        // This is a placeholder for actual evaluation
        println!("Evaluating example {}/{}", i + 1, test_examples.len());
        
        // Example: Generate response and compare with expected output
        let output = engine.generate(
            vec![example.clone()],
            Default::default()
        ).await?;
        
        // Add your evaluation metric here
        if output[0].text.contains("expected") {
            correct += 1;
        }
    }
    
    Ok(correct as f64 / test_examples.len() as f64)
}
```

## 📈 Advanced Techniques

### Mixed Precision Training
Enable FP16 or BF16 training for better performance:
```rust
let config = Config::new("Qwen/Qwen2-0.5B-Instruct")
    .with_device("cuda")
    .with_dtype("float16")  // or "bfloat16"
    .with_training(true);
```

### Gradient Accumulation
Train with larger effective batch sizes:
```rust
let training_config = TrainingConfig {
    // ... other configs ...
    gradient_accumulation_steps: 4,  // Effective batch size = batch_size * gradient_accumulation_steps
};
```

### Learning Rate Schedulers
```rust
let training_config = TrainingConfig {
    // ... other configs ...
    lr_scheduler: LRScheduler::Cosine {
        num_warmup_steps: 100,
        num_training_steps: 1000,
    },
};
```

## 🧠 Best Practices

1. **Data Quality**
   - Clean and normalize your data
   - Balance your dataset
   - Use data augmentation techniques

2. **Model Selection**
   - Start with a base model close to your domain
   - Consider model size vs. available resources

3. **Hyperparameter Tuning**
   - Use a validation set
   - Try different learning rates
   - Experiment with batch sizes

4. **Monitoring**
   - Track training metrics
   - Monitor GPU memory usage
   - Save checkpoints regularly

## 📚 Additional Resources

- [HuggingFace Transformers Documentation](https://huggingface.co/docs/transformers/)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)

## 🛠️ Troubleshooting

### Common Issues

1. **Out of Memory (OOM) Errors**
   - Reduce batch size
   - Use gradient accumulation
   - Enable gradient checkpointing

2. **Slow Training**
   - Enable mixed precision training
   - Use a larger batch size if memory allows
   - Consider using a smaller model

3. **Poor Performance**
   - Check your data quality
   - Try different learning rates
   - Increase training data size

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
