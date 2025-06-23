//! Scheduler for managing sequence execution and batching
//! 
//! This module implements the core scheduling logic that decides which sequences
//! to process, handles dynamic batching, and manages memory pressure through
//! intelligent preemption strategies.

use std::collections::VecDeque;
use crate::config::Config;
use crate::engine::sequence::{Sequence, SequenceStatus};
use crate::engine::block_manager::BlockManager;

/// Scheduler for managing sequence execution and batching
#[derive(Debug)]
pub struct Scheduler {
    /// Maximum number of sequences that can be processed simultaneously
    max_num_seqs: usize,
    
    /// Maximum number of tokens that can be batched together
    max_num_batched_tokens: usize,
    
    /// End-of-sequence token ID
    eos_token_id: Option<i64>,
    
    /// Block manager for KV cache management
    block_manager: BlockManager,
    
    /// Queue of sequences waiting to be scheduled
    waiting: VecDeque<Sequence>,
    
    /// Queue of sequences currently running
    running: VecDeque<Sequence>,
    
    /// Statistics for monitoring
    stats: SchedulerStats,
}

/// Statistics about scheduler performance
#[derive(Debug, Clone, Default)]
pub struct SchedulerStats {
    /// Total number of sequences processed
    pub total_sequences: usize,
    
    /// Number of sequences currently waiting
    pub waiting_sequences: usize,
    
    /// Number of sequences currently running
    pub running_sequences: usize,
    
    /// Number of sequences that have finished
    pub finished_sequences: usize,
    
    /// Number of preemptions that have occurred
    pub preemptions: usize,
    
    /// Total number of prefill batches processed
    pub prefill_batches: usize,
    
    /// Total number of decode batches processed
    pub decode_batches: usize,
    
    /// Average batch size for prefill
    pub avg_prefill_batch_size: f64,
    
    /// Average batch size for decode
    pub avg_decode_batch_size: f64,
}

impl Scheduler {
    /// Create a new scheduler
    pub fn new(config: &Config) -> Self {
        let block_manager = BlockManager::new(
            config.num_kvcache_blocks.unwrap_or(1000), // Default if not specified
            config.kvcache_block_size,
        );
        
        Self {
            max_num_seqs: config.max_num_seqs,
            max_num_batched_tokens: config.max_num_batched_tokens,
            eos_token_id: config.eos_token_id,
            block_manager,
            waiting: VecDeque::new(),
            running: VecDeque::new(),
            stats: SchedulerStats::default(),
        }
    }
    
    /// Check if all sequences are finished
    pub fn is_finished(&self) -> bool {
        self.waiting.is_empty() && self.running.is_empty()
    }
    
    /// Add a new sequence to the waiting queue
    pub fn add_sequence(&mut self, mut seq: Sequence) {
        seq.status = SequenceStatus::Waiting;
        self.waiting.push_back(seq);
        self.stats.total_sequences += 1;
        self.update_stats();
    }
    
    /// Schedule sequences for execution
    /// 
    /// Returns a tuple of (sequences_to_execute, is_prefill_batch)
    pub fn schedule(&mut self) -> anyhow::Result<(Vec<Sequence>, bool)> {
        // Try to schedule prefill batch first
        if let Some((sequences, true)) = self.try_schedule_prefill()? {
            self.stats.prefill_batches += 1;
            self.update_prefill_stats(sequences.len());
            return Ok((sequences, true));
        }
        
        // If no prefill batch possible, schedule decode batch
        let (sequences, false) = self.try_schedule_decode()?;
        self.stats.decode_batches += 1;
        self.update_decode_stats(sequences.len());
        Ok((sequences, false))
    }
    
    /// Try to schedule a prefill batch
    fn try_schedule_prefill(&mut self) -> anyhow::Result<Option<(Vec<Sequence>, bool)>> {
        if self.waiting.is_empty() {
            return Ok(None);
        }
        
        let mut scheduled_seqs = Vec::new();
        let mut num_seqs = 0;
        let mut num_batched_tokens = 0;
        
        // Process waiting sequences
        while let Some(seq) = self.waiting.front() {
            // Check batch size limits
            if num_seqs >= self.max_num_seqs {
                break;
            }
            
            let seq_tokens = seq.len() - seq.num_cached_tokens;
            if num_batched_tokens + seq_tokens > self.max_num_batched_tokens {
                break;
            }
            
            // Check if we can allocate memory for this sequence
            if !self.block_manager.can_allocate(seq) {
                break;
            }
            
            // Remove from waiting queue and prepare for execution
            let mut seq = self.waiting.pop_front().unwrap();
            
            // Allocate blocks for the sequence
            self.block_manager.allocate(&mut seq)?;
            
            num_seqs += 1;
            num_batched_tokens += seq_tokens;
            seq.status = SequenceStatus::Running;
            
            scheduled_seqs.push(seq);
        }
        
        if scheduled_seqs.is_empty() {
            return Ok(None);
        }
        
        // Move scheduled sequences to running queue
        for seq in &scheduled_seqs {
            self.running.push_back(seq.clone());
        }
        
        Ok(Some((scheduled_seqs, true)))
    }
    
    /// Try to schedule a decode batch
    fn try_schedule_decode(&mut self) -> anyhow::Result<(Vec<Sequence>, bool)> {
        let mut scheduled_seqs = Vec::new();
        let mut num_seqs = 0;
        
        // Process running sequences
        let mut sequences_to_reschedule = Vec::new();
        
        while let Some(seq) = self.running.pop_front() {
            if num_seqs >= self.max_num_seqs {
                sequences_to_reschedule.push(seq);
                continue;
            }
            
            // Check if we can append to this sequence (may need new block)
            while !self.block_manager.can_append(&seq) {
                if let Some(victim) = self.running.pop_back() {
                    // Preempt the last sequence to free up memory
                    self.preempt_sequence(victim);
                } else if !scheduled_seqs.is_empty() {
                    // Preempt from already scheduled sequences
                    let victim = scheduled_seqs.pop().unwrap();
                    self.preempt_sequence(victim);
                } else {
                    // Can't free up memory, preempt this sequence too
                    self.preempt_sequence(seq);
                    break;
                }
            }
            
            // If we successfully handled memory pressure, schedule this sequence
            if self.block_manager.can_append(&seq) {
                num_seqs += 1;
                self.block_manager.may_append(&seq)?;
                scheduled_seqs.push(seq);
            }
        }
        
        // Put back sequences that couldn't be scheduled
        for seq in sequences_to_reschedule.into_iter().rev() {
            self.running.push_front(seq);
        }
        
        // Put scheduled sequences back at the front of running queue
        for seq in scheduled_seqs.iter().rev() {
            self.running.push_front(seq.clone());
        }
        
        if scheduled_seqs.is_empty() {
            anyhow::bail!("No sequences could be scheduled for decode");
        }
        
        Ok((scheduled_seqs, false))
    }
    
    /// Preempt a sequence (move it back to waiting queue)
    fn preempt_sequence(&mut self, mut seq: Sequence) {
        seq.status = SequenceStatus::Preempted;
        self.block_manager.deallocate(&mut seq);
        self.waiting.push_front(seq);
        self.stats.preemptions += 1;
    }
    
    /// Process the results of sequence execution
    pub fn postprocess(&mut self, sequences: Vec<Sequence>, token_ids: Vec<i64>) -> anyhow::Result<()> {
        if sequences.len() != token_ids.len() {
            anyhow::bail!("Mismatch between sequences and token_ids length");
        }
        
        for (mut seq, token_id) in sequences.into_iter().zip(token_ids.into_iter()) {
            // Append the new token
            seq.append_token(token_id);
            
            // Check if sequence should finish
            if seq.should_stop(self.eos_token_id) {
                seq.finish();
                self.block_manager.deallocate(&mut seq);
                self.remove_from_running(&seq);
                self.stats.finished_sequences += 1;
            } else {
                // Update the sequence in the running queue
                self.update_running_sequence(seq);
            }
        }
        
        self.update_stats();
        Ok(())
    }
    
    /// Remove a sequence from the running queue
    fn remove_from_running(&mut self, target_seq: &Sequence) {
        self.running.retain(|seq| seq.seq_id != target_seq.seq_id);
    }
    
    /// Update a sequence in the running queue
    fn update_running_sequence(&mut self, updated_seq: Sequence) {
        for seq in &mut self.running {
            if seq.seq_id == updated_seq.seq_id {
                *seq = updated_seq;
                return;
            }
        }
        // If not found, add it back (shouldn't happen in normal operation)
        self.running.push_back(updated_seq);
    }
    
    /// Update general statistics
    fn update_stats(&mut self) {
        self.stats.waiting_sequences = self.waiting.len();
        self.stats.running_sequences = self.running.len();
    }
    
    /// Update prefill batch statistics
    fn update_prefill_stats(&mut self, batch_size: usize) {
        let total_batches = self.stats.prefill_batches as f64;
        let current_avg = self.stats.avg_prefill_batch_size;
        self.stats.avg_prefill_batch_size = 
            (current_avg * (total_batches - 1.0) + batch_size as f64) / total_batches;
    }
    
    /// Update decode batch statistics
    fn update_decode_stats(&mut self, batch_size: usize) {
        let total_batches = self.stats.decode_batches as f64;
        let current_avg = self.stats.avg_decode_batch_size;
        self.stats.avg_decode_batch_size = 
            (current_avg * (total_batches - 1.0) + batch_size as f64) / total_batches;
    }
    
    /// Get current scheduler statistics
    pub fn get_stats(&self) -> SchedulerStats {
        self.stats.clone()
    }
    
    /// Get block manager statistics
    pub fn get_block_stats(&self) -> crate::engine::block_manager::BlockManagerStats {
        self.block_manager.get_stats()
    }
    
    /// Get the number of sequences in each state
    pub fn get_queue_lengths(&self) -> (usize, usize) {
        (self.waiting.len(), self.running.len())
    }
    
    /// Force preempt all running sequences (for emergency memory cleanup)
    pub fn preempt_all(&mut self) {
        let running_seqs: Vec<_> = self.running.drain(..).collect();
        for seq in running_seqs {
            self.preempt_sequence(seq);
        }
    }
    
    /// Get memory pressure indicator (0.0 = no pressure, 1.0 = maximum pressure)
    pub fn memory_pressure(&self) -> f64 {
        let block_stats = self.block_manager.get_stats();
        if block_stats.total_blocks == 0 {
            0.0
        } else {
            1.0 - (block_stats.free_blocks as f64 / block_stats.total_blocks as f64)
        }
    }
    
    /// Check if the scheduler is under memory pressure
    pub fn is_under_memory_pressure(&self) -> bool {
        self.memory_pressure() > 0.9 // 90% memory usage threshold
    }
}

impl SchedulerStats {
    /// Calculate overall throughput (sequences per batch)
    pub fn overall_throughput(&self) -> f64 {
        let total_batches = self.prefill_batches + self.decode_batches;
        if total_batches == 0 {
            0.0
        } else {
            self.finished_sequences as f64 / total_batches as f64
        }
    }
    
    /// Calculate preemption rate
    pub fn preemption_rate(&self) -> f64 {
        if self.total_sequences == 0 {
            0.0
        } else {
            self.preemptions as f64 / self.total_sequences as f64
        }
    }
    
    /// Get completion rate
    pub fn completion_rate(&self) -> f64 {
        if self.total_sequences == 0 {
            0.0
        } else {
            self.finished_sequences as f64 / self.total_sequences as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::sampling_params::SamplingParams;
    
    fn create_test_config() -> Config {
        Config {
            model_path: std::path::PathBuf::from("/tmp"),
            max_num_batched_tokens: 1000,
            max_num_seqs: 10,
            max_model_len: 2048,
            gpu_memory_utilization: 0.9,
            tensor_parallel_size: 1,
            enforce_eager: false,
            eos_token_id: Some(2),
            kvcache_block_size: 16, // Small for testing
            num_kvcache_blocks: Some(100),
            device: "cpu".to_string(),
            dtype: "float32".to_string(),
        }
    }
    
    #[test]
    fn test_scheduler_creation() {
        let config = create_test_config();
        let scheduler = Scheduler::new(&config);
        
        assert_eq!(scheduler.max_num_seqs, 10);
        assert_eq!(scheduler.max_num_batched_tokens, 1000);
        assert_eq!(scheduler.eos_token_id, Some(2));
        assert!(scheduler.is_finished());
    }
    
    #[test]
    fn test_add_sequence() {
        let config = create_test_config();
        let mut scheduler = Scheduler::new(&config);
        
        let prompt_tokens = vec![1, 2, 3, 4, 5];
        let sampling_params = SamplingParams::new();
        let seq = Sequence::new(prompt_tokens, sampling_params);
        
        scheduler.add_sequence(seq);
        
        assert!(!scheduler.is_finished());
        assert_eq!(scheduler.waiting.len(), 1);
        assert_eq!(scheduler.stats.total_sequences, 1);
    }
    
    #[test]
    fn test_prefill_scheduling() {
        let config = create_test_config();
        let mut scheduler = Scheduler::new(&config);
        
        // Add a few sequences
        for i in 0..3 {
            let prompt_tokens = vec![1, 2, 3, 4, i];
            let sampling_params = SamplingParams::new();
            let seq = Sequence::new(prompt_tokens, sampling_params);
            scheduler.add_sequence(seq);
        }
        
        let (scheduled_seqs, is_prefill) = scheduler.schedule().unwrap();
        
        assert!(is_prefill);
        assert_eq!(scheduled_seqs.len(), 3);
        assert_eq!(scheduler.running.len(), 3);
        assert_eq!(scheduler.waiting.len(), 0);
        
        for seq in &scheduled_seqs {
            assert_eq!(seq.status, SequenceStatus::Running);
        }
    }
    
    #[test]
    fn test_decode_scheduling() {
        let config = create_test_config();
        let mut scheduler = Scheduler::new(&config);
        
        // Add and schedule a sequence first
        let prompt_tokens = vec![1, 2, 3, 4, 5];
        let sampling_params = SamplingParams::new();
        let seq = Sequence::new(prompt_tokens, sampling_params);
        scheduler.add_sequence(seq);
        
        // Schedule prefill
        let (prefill_seqs, _) = scheduler.schedule().unwrap();
        
        // Process prefill results
        let token_ids = vec![6]; // New token for the sequence
        scheduler.postprocess(prefill_seqs, token_ids).unwrap();
        
        // Now schedule decode
        let (decode_seqs, is_prefill) = scheduler.schedule().unwrap();
        
        assert!(!is_prefill);
        assert_eq!(decode_seqs.len(), 1);
        assert_eq!(decode_seqs[0].len(), 6); // Original 5 + 1 new token
    }
    
    #[test]
    fn test_sequence_completion() {
        let config = create_test_config();
        let mut scheduler = Scheduler::new(&config);
        
        let prompt_tokens = vec![1, 2, 3];
        let sampling_params = SamplingParams::new().with_max_tokens(1); // Only 1 completion token
        let seq = Sequence::new(prompt_tokens, sampling_params);
        scheduler.add_sequence(seq);
        
        // Schedule and process
        let (seqs, _) = scheduler.schedule().unwrap();
        let token_ids = vec![4]; // Add one token
        scheduler.postprocess(seqs, token_ids).unwrap();
        
        // Should be finished now due to max_tokens limit
        assert!(scheduler.is_finished());
        assert_eq!(scheduler.stats.finished_sequences, 1);
    }
    
    #[test]
    fn test_eos_token_completion() {
        let config = create_test_config();
        let mut scheduler = Scheduler::new(&config);
        
        let prompt_tokens = vec![1, 2, 3];
        let sampling_params = SamplingParams::new().with_max_tokens(10);
        let seq = Sequence::new(prompt_tokens, sampling_params);
        scheduler.add_sequence(seq);
        
        // Schedule and process with EOS token
        let (seqs, _) = scheduler.schedule().unwrap();
        let token_ids = vec![2]; // EOS token
        scheduler.postprocess(seqs, token_ids).unwrap();
        
        // Should be finished due to EOS token
        assert!(scheduler.is_finished());
        assert_eq!(scheduler.stats.finished_sequences, 1);
    }
    
    #[test]
    fn test_batch_size_limits() {
        let mut config = create_test_config();
        config.max_num_seqs = 2; // Limit to 2 sequences
        config.max_num_batched_tokens = 10; // Small token limit
        
        let mut scheduler = Scheduler::new(&config);
        
        // Add sequences that would exceed limits
        for i in 0..5 {
            let prompt_tokens = vec![1, 2, 3, 4, 5, 6]; // 6 tokens each
            let sampling_params = SamplingParams::new();
            let seq = Sequence::new(prompt_tokens, sampling_params);
            scheduler.add_sequence(seq);
        }
        
        let (scheduled_seqs, _) = scheduler.schedule().unwrap();
        
        // Should only schedule 1 sequence due to token limit (6 tokens < 10, but 12 > 10)
        assert_eq!(scheduled_seqs.len(), 1);
        assert_eq!(scheduler.waiting.len(), 4);
    }
    
    #[test]
    fn test_memory_pressure() {
        let mut config = create_test_config();
        config.num_kvcache_blocks = Some(5); // Very limited memory
        
        let mut scheduler = Scheduler::new(&config);
        
        // Add sequences that will cause memory pressure
        for i in 0..3 {
            let prompt_tokens = vec![1; 20]; // Large sequences
            let sampling_params = SamplingParams::new();
            let seq = Sequence::new(prompt_tokens, sampling_params);
            scheduler.add_sequence(seq);
        }
        
        // Try to schedule - should hit memory limits
        let result = scheduler.schedule();
        
        // Should either succeed with fewer sequences or handle gracefully
        assert!(result.is_ok());
        
        // Check memory pressure
        assert!(scheduler.memory_pressure() > 0.0);
    }
    
    #[test]
    fn test_statistics() {
        let config = create_test_config();
        let mut scheduler = Scheduler::new(&config);
        
        // Add and process some sequences
        for i in 0..3 {
            let prompt_tokens = vec![1, 2, 3, i];
            let sampling_params = SamplingParams::new().with_max_tokens(1);
            let seq = Sequence::new(prompt_tokens, sampling_params);
            scheduler.add_sequence(seq);
        }
        
        let (seqs, _) = scheduler.schedule().unwrap();
        let token_ids = vec![4, 5, 6];
        scheduler.postprocess(seqs, token_ids).unwrap();
        
        let stats = scheduler.get_stats();
        assert_eq!(stats.total_sequences, 3);
        assert_eq!(stats.finished_sequences, 3);
        assert_eq!(stats.prefill_batches, 1);
        assert_eq!(stats.avg_prefill_batch_size, 3.0);
        assert_eq!(stats.completion_rate(), 1.0);
    }
}