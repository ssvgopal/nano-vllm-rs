# nano-vllm-rs Examples

This directory contains example scripts demonstrating how to use the `nano-vllm-rs` library.

## Basic Demo

The `basic_demo.rs` script shows how to:
- Initialize the LLM engine
- Configure generation parameters
- Generate text from prompts
- Monitor performance and memory usage

### Running the Example

1. Make sure you have Rust and Cargo installed
2. Install the required system dependencies (CUDA if using GPU)
3. Run the example with:

```bash
# Run with default CPU backend
cargo run --example basic_demo --features="cpu"

# Or with CUDA support (if available)
cargo run --example basic_demo --features="cuda"
```

### Requirements

- A compatible model in the Hugging Face format
- Sufficient system resources (RAM/VRAM) for the model
- Optional: CUDA for GPU acceleration

### Expected Output

The demo will:
1. Initialize the engine with the specified model
2. Display engine statistics
3. Generate text for the provided prompts
4. Show performance metrics and memory usage

## Adding More Examples

To add a new example:
1. Create a new `.rs` file in this directory
2. Add your example code
3. Update this README with instructions for your example
4. Test it with `cargo run --example your_example_name`
