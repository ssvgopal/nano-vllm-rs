//! Test utilities for nano-vllm-rs

use std::sync::Once;
use tracing_subscriber::{fmt, EnvFilter};

static INIT: Once = Once::new();

/// Initialize test logger
pub fn init_test_logging() {
    INIT.call_once(|| {
        // Set default log level to info if not specified
        if std::env::var_os("RUST_LOG").is_none() {
            std::env::set_var("RUST_LOG", "info");
        }
        
        // Initialize logging
        fmt()
            .with_env_filter(EnvFilter::from_default_env())
            .with_test_writer()
            .init();
            
        // Log test start
        tracing::info!("Initialized test logging");
    });
}

/// Get the test model path from environment or use default
pub fn test_model() -> String {
    std::env::var("TEST_MODEL").unwrap_or_else(|_| "Qwen/Qwen2-0.5B-Instruct".to_string())
}

/// Create a test engine with default settings
pub async fn create_test_engine() -> crate::LLMEngine {
    init_test_logging();
    
    crate::LLMEngineBuilder::new()
        .model_path(test_model())
        .device("cpu")
        .max_num_seqs(4)
        .max_num_batched_tokens(2048)
        .build()
        .await
        .expect("Failed to create test engine")
}

/// Assert that two floating point numbers are approximately equal
#[macro_export]
macro_rules! assert_approx_eq {
    ($a:expr, $b:expr, $eps:expr) => {
        assert!(
            ($a - $b).abs() < $eps,
            "assertion failed: |{} - {}| < {}",
            $a,
            $b,
            $eps
        );
    };
    ($a:expr, $b:expr) => {
        assert_approx_eq!($a, $b, 1e-6);
    };
}

/// Assert that a result is an error containing the expected message
#[macro_export]
macro_rules! assert_error_contains {
    ($result:expr, $msg:expr) => {
        match $result {
            Ok(_) => panic!("Expected error containing '{}', but got Ok", $msg),
            Err(e) => {
                let err_str = e.to_string();
                assert!(
                    err_str.contains($msg),
                    "Expected error to contain '{}', but got: {}",
                    $msg,
                    err_str
                );
            }
        }
    };
}
