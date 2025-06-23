# nano-vllm-rs Test Suite

This directory contains the test suite for `nano-vllm-rs`, organized into different levels of testing.

## Test Organization

- `unit/`: Unit tests for individual components
  - `sampling_test.rs`: Tests for sampling algorithms (top-k, top-p, temperature, etc.)
  - `model_test.rs`: Tests for model components and configurations

- `integration/`: Integration tests for the engine and its components
  - `engine_test.rs`: Tests for the main LLM engine functionality

- `e2e/`: End-to-end tests for the complete pipeline
  - `pipeline_test.rs`: Tests the complete text generation pipeline

- `test_utils.rs`: Common test utilities and helpers

## Running Tests

### Run All Tests

```bash
cargo test --all-features
```

### Run Specific Test Category

```bash
# Unit tests
cargo test --test unit

# Integration tests
cargo test --test integration

# End-to-end tests
cargo test --test e2e
```

### Run with Logging

```bash
# Set log level (error, warn, info, debug, trace)
RUST_LOG=info cargo test -- --nocapture
```

### Run with Specific Model

By default, tests use a small test model. To use a different model:

```bash
TEST_MODEL=your/model/path cargo test
```

## Writing Tests

1. **Unit Tests**: Test individual functions and methods in isolation.
2. **Integration Tests**: Test interactions between components.
3. **End-to-End Tests**: Test the complete pipeline from input to output.

### Test Utilities

Use the utilities in `test_utils.rs` for common test functionality:

```rust
use test_utils::{init_test_logging, test_model, create_test_engine};

#[test]
fn test_example() {
    init_test_logging();
    // Test code here
}
```

## Test Coverage

To generate test coverage reports:

```bash
# Install cargo-tarpaulin if needed
cargo install cargo-tarpaulin

# Run coverage
cargo tarpaulin --workspace --ignore-tests --out Html --output-dir ./target/coverage
```

## Performance Testing

For performance benchmarks, see the `benches/` directory.

## Continuous Integration

Tests are automatically run on CI for all pull requests. See `.github/workflows/ci.yml` for details.
