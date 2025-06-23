//! Sequence management for nano-vLLM
//! 
//! This module defines the core Sequence type that represents an individual
//! inference request, tracking its state, tokens, and metadata throughout
//! the generation process.

use std::sync::atomic::{AtomicU64, Ordering};
use serde::{Deserialize, Serialize};
use crate::engine::sampling_params::SamplingParams;

/// Global sequence ID counter
static SEQUENCE_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Status of a sequence during processing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SequenceStatus {
    /// Sequence is waiting to be scheduled
    Waiting,
    /// Sequence is currently being processed
    Running,
    /// Sequence has completed generation
    Finished,
    /// Sequence was preempted and needs to be rescheduled
    Preempted,
    /// Sequence encountered an error
    Error,
}

/// Output from a completed sequence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequenceOutput {
    /// Unique sequence identifier
    pub seq_id: u64,
    /// Generated text
    pub text: String,
    /// All token IDs (prompt + completion)
    pub token_ids: Vec<i64>,
    /// Only the completion token IDs
    pub completion_token_ids: Vec<i64>,
    /// Number of prompt tokens
    pub num_prompt_tokens: usize,
    /// Number of completion tokens
    pub num_completion_tokens: usize,
    /// Final status of the sequence
    pub status: SequenceStatus,
}

/// A sequence represents a single inference request
#[derive(Debug, Clone)]
pub struct Sequence {
    /// Unique sequence identifier
    pub seq_id: u64,
    
    /// Current status of the sequence
    pub status: SequenceStatus,
    
    /// All token IDs (prompt + generated tokens)
    pub token_ids: Vec<i64>,
    
    /// The last generated token
    pub last_token: i64,
    
    /// Total number of tokens in the sequence
    pub num_tokens: usize,
    
    /// Number of prompt tokens (immutable)
    pub num_prompt_tokens: usize,
    
    /// Number of tokens that are cached (for prefix caching)
    pub num_cached_tokens: usize,
    
    /// Block table for KV cache management
    pub block_table: Vec<i32>,
    
    /// Sampling parameters for this sequence
    pub sampling_params: SamplingParams,
    
    /// Block size for KV cache (typically 256)
    pub block_size: usize,
}

impl Sequence {
    /// Create a new sequence from prompt tokens and sampling parameters
    pub fn new(prompt_token_ids: Vec<i64>, sampling_params: SamplingParams) -> Self {
        let seq_id = SEQUENCE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let num_prompt_tokens = prompt_token_ids.len();
        let last_token = *prompt_token_ids.last().unwrap_or(&0);
        
        Self {
            seq_id,
            status: SequenceStatus::Waiting,
            token_ids: prompt_token_ids,
            last_token,
            num_tokens: num_prompt_tokens,
            num_prompt_tokens,
            num_cached_tokens: 0,
            block_table: Vec::new(),
            sampling_params,
            block_size: 256, // Standard block size
        }
    }
    
    /// Get the length of the sequence (total tokens)
    pub fn len(&self) -> usize {
        self.num_tokens
    }
    
    /// Check if the sequence is empty
    pub fn is_empty(&self) -> bool {
        self.num_tokens == 0
    }
    
    /// Get a slice of token IDs
    pub fn get_token_ids(&self, start: usize, end: Option<usize>) -> &[i64] {
        let end = end.unwrap_or(self.num_tokens);
        &self.token_ids[start..end.min(self.token_ids.len())]
    }
    
    /// Get all token IDs
    pub fn all_token_ids(&self) -> &[i64] {
        &self.token_ids
    }
    
    /// Get prompt token IDs
    pub fn prompt_token_ids(&self) -> &[i64] {
        &self.token_ids[..self.num_prompt_tokens]
    }
    
    /// Get completion token IDs
    pub fn completion_token_ids(&self) -> Vec<i64> {
        self.token_ids[self.num_prompt_tokens..].to_vec()
    }
    
    /// Get the number of completion tokens generated
    pub fn num_completion_tokens(&self) -> usize {
        self.num_tokens - self.num_prompt_tokens
    }
    
    /// Check if the sequence is finished
    pub fn is_finished(&self) -> bool {
        matches!(self.status, SequenceStatus::Finished | SequenceStatus::Error)
    }
    
    /// Check if the sequence can be scheduled
    pub fn can_schedule(&self) -> bool {
        matches!(self.status, SequenceStatus::Waiting | SequenceStatus::Preempted)
    }
    
    /// Append a new token to the sequence
    pub fn append_token(&mut self, token_id: i64) {
        self.token_ids.push(token_id);
        self.last_token = token_id;
        self.num_tokens += 1;
    }
    
    /// Get the number of blocks needed for this sequence
    pub fn num_blocks(&self) -> usize {
        (self.num_tokens + self.block_size - 1) / self.block_size
    }
    
    /// Get the number of cached blocks
    pub fn num_cached_blocks(&self) -> usize {
        self.num_cached_tokens / self.block_size
    }
    
    /// Get the number of tokens in the last block
    pub fn last_block_num_tokens(&self) -> usize {
        let remainder = self.num_tokens % self.block_size;
        if remainder == 0 && self.num_tokens > 0 {
            self.block_size
        } else {
            remainder
        }
    }
    
    /// Get tokens for a specific block
    pub fn get_block_tokens(&self, block_idx: usize) -> Vec<i64> {
        let start = block_idx * self.block_size;
        let end = ((block_idx + 1) * self.block_size).min(self.num_tokens);
        
        if start >= self.num_tokens {
            Vec::new()
        } else {
            self.token_ids[start..end].to_vec()
        }
    }
    
    /// Check if generation should stop
    pub fn should_stop(&self, eos_token_id: Option<i64>) -> bool {
        // Check max tokens limit
        if self.num_completion_tokens() >= self.sampling_params.max_tokens {
            return true;
        }
        
        // Check EOS token (if not ignoring EOS)
        if !self.sampling_params.ignore_eos {
            if let Some(eos_id) = eos_token_id {
                if self.last_token == eos_id {
                    return true;
                }
            }
        }
        
        false
    }
    
    /// Mark the sequence as finished
    pub fn finish(&mut self) {
        self.status = SequenceStatus::Finished;
    }
    
    /// Mark the sequence as preempted
    pub fn preempt(&mut self) {
        self.status = SequenceStatus::Preempted;
        // Clear block table and cached tokens when preempted
        self.block_table.clear();
        self.num_cached_tokens = 0;
    }
    
    /// Mark the sequence as running
    pub fn set_running(&mut self) {
        self.status = SequenceStatus::Running;
    }
    
    /// Create a sequence output for completed sequences
    pub fn create_output(&self, text: String) -> SequenceOutput {
        SequenceOutput {
            seq_id: self.seq_id,
            text,
            token_ids: self.token_ids.clone(),
            completion_token_ids: self.completion_token_ids(),
            num_prompt_tokens: self.num_prompt_tokens,
            num_completion_tokens: self.num_completion_tokens(),
            status: self.status,
        }
    }
}

impl std::fmt::Display for Sequence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Sequence(id={}, status={:?}, tokens={}, prompt={}, completion={})",
            self.seq_id,
            self.status,
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_completion_tokens()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sequence_creation() {
        let prompt_tokens = vec![1, 2, 3, 4, 5];
        let sampling_params = SamplingParams::new().with_max_tokens(10);
        let seq = Sequence::new(prompt_tokens.clone(), sampling_params);
        
        assert_eq!(seq.num_prompt_tokens, 5);
        assert_eq!(seq.num_tokens, 5);
        assert_eq!(seq.last_token, 5);
        assert_eq!(seq.prompt_token_ids(), &prompt_tokens);
        assert_eq!(seq.num_completion_tokens(), 0);
        assert!(!seq.is_finished());
        assert!(seq.can_schedule());
    }
    
    #[test]
    fn test_token_appending() {
        let prompt_tokens = vec![1, 2, 3];
        let sampling_params = SamplingParams::new();
        let mut seq = Sequence::new(prompt_tokens, sampling_params);
        
        seq.append_token(4);
        seq.append_token(5);
        
        assert_eq!(seq.num_tokens, 5);
        assert_eq!(seq.num_completion_tokens(), 2);
        assert_eq!(seq.last_token, 5);
        assert_eq!(seq.completion_token_ids(), vec![4, 5]);
    }
    
    #[test]
    fn test_block_calculations() {
        let prompt_tokens = vec![1; 300]; // 300 tokens
        let sampling_params = SamplingParams::new();
        let seq = Sequence::new(prompt_tokens, sampling_params);
        
        // With block size 256, we need 2 blocks for 300 tokens
        assert_eq!(seq.num_blocks(), 2);
        assert_eq!(seq.last_block_num_tokens(), 44); // 300 - 256 = 44
        
        // Test block token retrieval
        let block_0_tokens = seq.get_block_tokens(0);
        assert_eq!(block_0_tokens.len(), 256);
        
        let block_1_tokens = seq.get_block_tokens(1);
        assert_eq!(block_1_tokens.len(), 44);
    }
    
    #[test]
    fn test_should_stop() {
        let prompt_tokens = vec![1, 2, 3];
        let sampling_params = SamplingParams::new().with_max_tokens(2);
        let mut seq = Sequence::new(prompt_tokens, sampling_params);
        
        // Should not stop initially
        assert!(!seq.should_stop(Some(999)));
        
        // Add tokens up to max_tokens
        seq.append_token(4);
        seq.append_token(5);
        
        // Should stop due to max tokens
        assert!(seq.should_stop(Some(999)));
        
        // Test EOS stopping
        let mut seq2 = Sequence::new(vec![1, 2, 3], SamplingParams::new().with_max_tokens(10));
        seq2.append_token(999); // EOS token
        assert!(seq2.should_stop(Some(999)));
        
        // Test ignoring EOS
        let mut seq3 = Sequence::new(
            vec![1, 2, 3], 
            SamplingParams::new().with_max_tokens(10).with_ignore_eos(true)
        );
        seq3.append_token(999); // EOS token
        assert!(!seq3.should_stop(Some(999))); // Should not stop because ignore_eos is true
    }
    
    #[test]
    fn test_sequence_status_transitions() {
        let prompt_tokens = vec![1, 2, 3];
        let sampling_params = SamplingParams::new();
        let mut seq = Sequence::new(prompt_tokens, sampling_params);
        
        // Initial state
        assert_eq!(seq.status, SequenceStatus::Waiting);
        assert!(seq.can_schedule());
        
        // Set running
        seq.set_running();
        assert_eq!(seq.status, SequenceStatus::Running);
        assert!(!seq.can_schedule());
        
        // Preempt
        seq.preempt();
        assert_eq!(seq.status, SequenceStatus::Preempted);
        assert!(seq.can_schedule());
        assert!(seq.block_table.is_empty());
        assert_eq!(seq.num_cached_tokens, 0);
        
        // Finish
        seq.finish();
        assert_eq!(seq.status, SequenceStatus::Finished);
        assert!(!seq.can_schedule());
        assert!(seq.is_finished());
    }
}