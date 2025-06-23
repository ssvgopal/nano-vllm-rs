//! Context management for inference operations
//! 
//! This module provides a global context system for managing state
//! during model inference, including attention metadata and memory mappings.

use std::sync::RwLock;
use candle_core::Tensor;

/// Context information for model inference operations
#[derive(Debug, Clone)]
pub struct Context {
    /// Whether this is a prefill operation (vs decode)
    pub is_prefill: bool,
    
    /// Cumulative sequence lengths for queries (prefill only)
    pub cu_seqlens_q: Option<Tensor>,
    
    /// Cumulative sequence lengths for keys (prefill only)
    pub cu_seqlens_k: Option<Tensor>,
    
    /// Maximum sequence length for queries
    pub max_seqlen_q: usize,
    
    /// Maximum sequence length for keys
    pub max_seqlen_k: usize,
    
    /// Slot mapping for KV cache storage
    pub slot_mapping: Option<Tensor>,
    
    /// Context lengths for each sequence (decode only)
    pub context_lens: Option<Tensor>,
    
    /// Block tables for attention computation
    pub block_tables: Option<Tensor>,
}

impl Default for Context {
    fn default() -> Self {
        Self {
            is_prefill: false,
            cu_seqlens_q: None,
            cu_seqlens_k: None,
            max_seqlen_q: 0,
            max_seqlen_k: 0,
            slot_mapping: None,
            context_lens: None,
            block_tables: None,
        }
    }
}

impl Context {
    /// Create a new empty context
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Create a prefill context
    pub fn prefill(
        cu_seqlens_q: Tensor,
        cu_seqlens_k: Tensor,
        max_seqlen_q: usize,
        max_seqlen_k: usize,
        slot_mapping: Tensor,
        block_tables: Option<Tensor>,
    ) -> Self {
        Self {
            is_prefill: true,
            cu_seqlens_q: Some(cu_seqlens_q),
            cu_seqlens_k: Some(cu_seqlens_k),
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping: Some(slot_mapping),
            context_lens: None,
            block_tables,
        }
    }
    
    /// Create a decode context
    pub fn decode(
        slot_mapping: Tensor,
        context_lens: Tensor,
        block_tables: Tensor,
    ) -> Self {
        Self {
            is_prefill: false,
            cu_seqlens_q: None,
            cu_seqlens_k: None,
            max_seqlen_q: 0,
            max_seqlen_k: 0,
            slot_mapping: Some(slot_mapping),
            context_lens: Some(context_lens),
            block_tables: Some(block_tables),
        }
    }
    
    /// Check if this context has prefix caching enabled
    pub fn has_prefix_cache(&self) -> bool {
        self.is_prefill && self.block_tables.is_some()
    }
    
    /// Get the batch size from the context
    pub fn batch_size(&self) -> usize {
        if self.is_prefill {
            if let Some(cu_seqlens_q) = &self.cu_seqlens_q {
                // Batch size is length - 1 for cumulative sequence lengths
                cu_seqlens_q.dims().get(0).map(|&d| d.saturating_sub(1)).unwrap_or(0)
            } else {
                0
            }
        } else {
            self.context_lens
                .as_ref()
                .and_then(|t| t.dims().get(0))
                .copied()
                .unwrap_or(0)
        }
    }
    
    /// Validate the context for consistency
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.is_prefill {
            // Prefill validation
            if self.cu_seqlens_q.is_none() || self.cu_seqlens_k.is_none() {
                anyhow::bail!("Prefill context missing cumulative sequence lengths");
            }
            
            if self.slot_mapping.is_none() {
                anyhow::bail!("Prefill context missing slot mapping");
            }
            
            if self.max_seqlen_q == 0 || self.max_seqlen_k == 0 {
                anyhow::bail!("Prefill context has zero max sequence lengths");
            }
        } else {
            // Decode validation
            if self.slot_mapping.is_none() {
                anyhow::bail!("Decode context missing slot mapping");
            }
            
            if self.context_lens.is_none() {
                anyhow::bail!("Decode context missing context lengths");
            }
            
            if self.block_tables.is_none() {
                anyhow::bail!("Decode context missing block tables");
            }
        }
        
        Ok(())
    }
}

/// Global context storage
static GLOBAL_CONTEXT: RwLock<Context> = RwLock::new(Context {
    is_prefill: false,
    cu_seqlens_q: None,
    cu_seqlens_k: None,
    max_seqlen_q: 0,
    max_seqlen_k: 0,
    slot_mapping: None,
    context_lens: None,
    block_tables: None,
});

/// Get the current global context
pub fn get_context() -> Context {
    GLOBAL_CONTEXT.read().unwrap().clone()
}

/// Set the global context
pub fn set_context(context: Context) -> anyhow::Result<()> {
    context.validate()?;
    *GLOBAL_CONTEXT.write().unwrap() = context;
    Ok(())
}

/// Set a prefill context
pub fn set_prefill_context(
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    slot_mapping: Tensor,
    block_tables: Option<Tensor>,
) -> anyhow::Result<()> {
    let context = Context::prefill(
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        slot_mapping,
        block_tables,
    );
    set_context(context)
}

/// Set a decode context
pub fn set_decode_context(
    slot_mapping: Tensor,
    context_lens: Tensor,
    block_tables: Tensor,
) -> anyhow::Result<()> {
    let context = Context::decode(slot_mapping, context_lens, block_tables);
    set_context(context)
}

/// Reset the global context to default
pub fn reset_context() {
    *GLOBAL_CONTEXT.write().unwrap() = Context::default();
}

/// Execute a function with a temporary context
pub fn with_context<F, R>(context: Context, f: F) -> anyhow::Result<R>
where
    F: FnOnce() -> R,
{
    let old_context = get_context();
    set_context(context)?;
    let result = f();
    set_context(old_context)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, DType};
    
    #[test]
    fn test_context_creation() {
        let context = Context::new();
        assert!(!context.is_prefill);
        assert!(context.cu_seqlens_q.is_none());
        assert!(context.slot_mapping.is_none());
    }
    
    #[test]
    fn test_prefill_context() {
        let device = Device::Cpu;
        let cu_seqlens_q = Tensor::from_slice(&[0i32, 5, 10], (3,), &device).unwrap();
        let cu_seqlens_k = Tensor::from_slice(&[0i32, 5, 10], (3,), &device).unwrap();
        let slot_mapping = Tensor::from_slice(&[0i32, 1, 2, 3, 4], (5,), &device).unwrap();
        
        let context = Context::prefill(
            cu_seqlens_q,
            cu_seqlens_k,
            5,
            5,
            slot_mapping,
            None,
        );
        
        assert!(context.is_prefill);
        assert_eq!(context.max_seqlen_q, 5);
        assert_eq!(context.max_seqlen_k, 5);
        assert_eq!(context.batch_size(), 2); // 3 - 1 = 2
        assert!(!context.has_prefix_cache());
    }
    
    #[test]
    fn test_decode_context() {
        let device = Device::Cpu;
        let slot_mapping = Tensor::from_slice(&[0i32, 1], (2,), &device).unwrap();
        let context_lens = Tensor::from_slice(&[5i32, 7], (2,), &device).unwrap();
        let block_tables = Tensor::from_slice(&[0i32, 1, 2, 3], (2, 2), &device).unwrap();
        
        let context = Context::decode(slot_mapping, context_lens, block_tables);
        
        assert!(!context.is_prefill);
        assert_eq!(context.batch_size(), 2);
        assert!(context.context_lens.is_some());
        assert!(context.block_tables.is_some());
    }
    
    #[test]
    fn test_global_context() {
        // Reset to ensure clean state
        reset_context();
        
        let initial_context = get_context();
        assert!(!initial_context.is_prefill);
        
        // Create and set a new context
        let device = Device::Cpu;
        let slot_mapping = Tensor::from_slice(&[0i32], (1,), &device).unwrap();
        let context_lens = Tensor::from_slice(&[5i32], (1,), &device).unwrap();
        let block_tables = Tensor::from_slice(&[0i32], (1, 1), &device).unwrap();
        
        let new_context = Context::decode(slot_mapping, context_lens, block_tables);
        set_context(new_context).unwrap();
        
        let retrieved_context = get_context();
        assert!(!retrieved_context.is_prefill);
        assert_eq!(retrieved_context.batch_size(), 1);
        
        // Reset again
        reset_context();
        let reset_context = get_context();
        assert!(!reset_context.is_prefill);
        assert_eq!(reset_context.batch_size(), 0);
    }
    
    #[test]
    fn test_with_context() {
        reset_context();
        
        let device = Device::Cpu;
        let slot_mapping = Tensor::from_slice(&[0i32], (1,), &device).unwrap();
        let context_lens = Tensor::from_slice(&[5i32], (1,), &device).unwrap();
        let block_tables = Tensor::from_slice(&[0i32], (1, 1), &device).unwrap();
        
        let temp_context = Context::decode(slot_mapping, context_lens, block_tables);
        
        let result = with_context(temp_context, || {
            let ctx = get_context();
            ctx.batch_size()
        }).unwrap();
        
        assert_eq!(result, 1);
        
        // Context should be reset to default after the closure
        let final_context = get_context();
        assert_eq!(final_context.batch_size(), 0);
    }
}