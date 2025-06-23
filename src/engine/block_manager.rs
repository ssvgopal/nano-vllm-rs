//! Block manager for KV cache memory management
//! 
//! This module implements the core memory management system for KV cache,
//! including prefix caching with hash-based block deduplication.

use std::collections::{HashMap, VecDeque};
use xxhash_rust::xxh64::xxh64;
use crate::engine::sequence::Sequence;

/// A single block in the KV cache
#[derive(Debug, Clone)]
pub struct Block {
    /// Unique block identifier
    pub block_id: usize,
    
    /// Reference count for this block
    pub ref_count: usize,
    
    /// Hash of the token content (for prefix caching)
    pub hash: Option<u64>,
    
    /// Token IDs stored in this block
    pub token_ids: Vec<i64>,
}

impl Block {
    /// Create a new block with the given ID
    pub fn new(block_id: usize) -> Self {
        Self {
            block_id,
            ref_count: 0,
            hash: None,
            token_ids: Vec::new(),
        }
    }
    
    /// Update the block with new hash and token IDs
    pub fn update(&mut self, hash: u64, token_ids: Vec<i64>) {
        self.hash = Some(hash);
        self.token_ids = token_ids;
    }
    
    /// Reset the block to initial state
    pub fn reset(&mut self) {
        self.ref_count = 1;
        self.hash = None;
        self.token_ids.clear();
    }
    
    /// Check if this block is free (ref_count == 0)
    pub fn is_free(&self) -> bool {
        self.ref_count == 0
    }
    
    /// Increment reference count
    pub fn add_ref(&mut self) {
        self.ref_count += 1;
    }
    
    /// Decrement reference count
    pub fn remove_ref(&mut self) {
        assert!(self.ref_count > 0, "Cannot remove reference from block with zero refs");
        self.ref_count -= 1;
    }
}

/// Block manager for KV cache memory management
#[derive(Debug)]
pub struct BlockManager {
    /// Total number of blocks available
    num_blocks: usize,
    
    /// Size of each block (number of tokens)
    block_size: usize,
    
    /// All blocks in the system
    blocks: Vec<Block>,
    
    /// Mapping from hash to block ID for prefix caching
    hash_to_block_id: HashMap<u64, usize>,
    
    /// Queue of free block IDs
    free_block_ids: VecDeque<usize>,
    
    /// Set of currently used block IDs
    used_block_ids: std::collections::HashSet<usize>,
}

impl BlockManager {
    /// Create a new block manager
    pub fn new(num_blocks: usize, block_size: usize) -> Self {
        assert!(num_blocks > 0, "Number of blocks must be positive");
        assert!(block_size > 0, "Block size must be positive");
        
        let blocks = (0..num_blocks).map(Block::new).collect();
        let free_block_ids = (0..num_blocks).collect();
        
        Self {
            num_blocks,
            block_size,
            blocks,
            hash_to_block_id: HashMap::new(),
            free_block_ids,
            used_block_ids: std::collections::HashSet::new(),
        }
    }
    
    /// Compute hash for a sequence of token IDs with optional prefix
    pub fn compute_hash(token_ids: &[i64], prefix_hash: Option<u64>) -> u64 {
        let mut data = Vec::new();
        
        // Include prefix hash if provided
        if let Some(prefix) = prefix_hash {
            data.extend_from_slice(&prefix.to_le_bytes());
        }
        
        // Add token IDs as bytes
        for &token_id in token_ids {
            data.extend_from_slice(&token_id.to_le_bytes());
        }
        
        xxh64(&data, 0)
    }
    
    /// Allocate a specific block
    fn allocate_block(&mut self, block_id: usize) -> &mut Block {
        assert!(self.blocks[block_id].is_free(), "Block {} is not free", block_id);
        
        self.blocks[block_id].reset();
        self.free_block_ids.retain(|&id| id != block_id);
        self.used_block_ids.insert(block_id);
        
        &mut self.blocks[block_id]
    }
    
    /// Deallocate a specific block
    fn deallocate_block(&mut self, block_id: usize) {
        assert!(self.blocks[block_id].is_free(), "Block {} still has references", block_id);
        
        self.used_block_ids.remove(&block_id);
        self.free_block_ids.push_back(block_id);
        
        // Remove from hash mapping if it exists
        if let Some(hash) = self.blocks[block_id].hash {
            if self.hash_to_block_id.get(&hash) == Some(&block_id) {
                self.hash_to_block_id.remove(&hash);
            }
        }
    }
    
    /// Check if we can allocate blocks for a sequence
    pub fn can_allocate(&self, seq: &Sequence) -> bool {
        self.free_block_ids.len() >= seq.num_blocks()
    }
    
    /// Allocate blocks for a sequence with prefix caching
    pub fn allocate(&mut self, seq: &mut Sequence) -> anyhow::Result<()> {
        if !seq.block_table.is_empty() {
            anyhow::bail!("Sequence already has allocated blocks");
        }
        
        if !self.can_allocate(seq) {
            anyhow::bail!("Not enough free blocks to allocate sequence");
        }
        
        let mut prefix_hash: Option<u64> = None;
        let mut cache_miss = false;
        
        for block_idx in 0..seq.num_blocks() {
            let block_tokens = seq.get_block_tokens(block_idx);
            
            // Only compute hash for full blocks
            let current_hash = if block_tokens.len() == self.block_size {
                Some(Self::compute_hash(&block_tokens, prefix_hash))
            } else {
                None
            };
            
            let block_id = if let Some(hash) = current_hash {
                // Try to find existing block with this hash
                if let Some(&existing_block_id) = self.hash_to_block_id.get(&hash) {
                    let existing_block = &self.blocks[existing_block_id];
                    
                    // Verify token content matches (hash collision protection)
                    if !cache_miss && existing_block.token_ids == block_tokens {
                        // Cache hit! Reuse existing block
                        seq.num_cached_tokens += self.block_size;
                        
                        if self.used_block_ids.contains(&existing_block_id) {
                            // Block is already in use, increment reference
                            self.blocks[existing_block_id].add_ref();
                            existing_block_id
                        } else {
                            // Block is free, allocate it
                            self.allocate_block(existing_block_id);
                            existing_block_id
                        }
                    } else {
                        // Cache miss due to content mismatch or previous miss
                        cache_miss = true;
                        self.allocate_new_block(hash, block_tokens)?
                    }
                } else {
                    // Hash not found, allocate new block
                    cache_miss = true;
                    self.allocate_new_block(hash, block_tokens)?
                }
            } else {
                // Partial block, always allocate new
                cache_miss = true;
                self.allocate_new_block(None, block_tokens)?
            };
            
            seq.block_table.push(block_id as i32);
            prefix_hash = current_hash;
        }
        
        Ok(())
    }
    
    /// Allocate a new block with given hash and tokens
    fn allocate_new_block(&mut self, hash: Option<u64>, token_ids: Vec<i64>) -> anyhow::Result<usize> {
        let block_id = self.free_block_ids.front()
            .copied()
            .ok_or_else(|| anyhow::anyhow!("No free blocks available"))?;
        
        let block = self.allocate_block(block_id);
        
        if let Some(hash) = hash {
            block.update(hash, token_ids);
            self.hash_to_block_id.insert(hash, block_id);
        } else {
            block.token_ids = token_ids;
        }
        
        Ok(block_id)
    }
    
    /// Deallocate all blocks for a sequence
    pub fn deallocate(&mut self, seq: &mut Sequence) {
        for &block_id in seq.block_table.iter().rev() {
            let block = &mut self.blocks[block_id as usize];
            block.remove_ref();
            
            if block.is_free() {
                self.deallocate_block(block_id as usize);
            }
        }
        
        seq.num_cached_tokens = 0;
        seq.block_table.clear();
    }
    
    /// Check if we can append a token to a sequence (may need new block)
    pub fn can_append(&self, seq: &Sequence) -> bool {
        // If we're at the start of a new block, we need a free block
        if seq.len() % self.block_size == 1 {
            !self.free_block_ids.is_empty()
        } else {
            true // Can always append within existing block
        }
    }
    
    /// Handle block allocation when appending tokens
    pub fn may_append(&mut self, seq: &mut Sequence) -> anyhow::Result<()> {
        if seq.block_table.is_empty() {
            anyhow::bail!("Sequence has no allocated blocks");
        }
        
        let last_block_idx = seq.block_table.len() - 1;
        let last_block_id = seq.block_table[last_block_idx] as usize;
        let last_block = &mut self.blocks[last_block_id];
        
        if seq.len() % self.block_size == 1 {
            // We just started a new block, need to allocate it
            if last_block.hash.is_some() {
                // Previous block was full and hashed
                let new_block_id = self.free_block_ids.front()
                    .copied()
                    .ok_or_else(|| anyhow::anyhow!("No free blocks for append"))?;
                
                self.allocate_block(new_block_id);
                seq.block_table.push(new_block_id as i32);
            }
        } else if seq.len() % self.block_size == 0 {
            // We just completed a block, compute its hash
            if last_block.hash.is_none() {
                let block_tokens = seq.get_block_tokens(seq.num_blocks() - 1);
                let prefix_hash = if seq.block_table.len() > 1 {
                    let prev_block_id = seq.block_table[last_block_idx - 1] as usize;
                    self.blocks[prev_block_id].hash
                } else {
                    None
                };
                
                let hash = Self::compute_hash(&block_tokens, prefix_hash);
                last_block.update(hash, block_tokens);
                self.hash_to_block_id.insert(hash, last_block_id);
            }
        }
        // For other cases (middle of block), no action needed
        
        Ok(())
    }
    
    /// Get statistics about block usage
    pub fn get_stats(&self) -> BlockManagerStats {
        BlockManagerStats {
            total_blocks: self.num_blocks,
            free_blocks: self.free_block_ids.len(),
            used_blocks: self.used_block_ids.len(),
            cached_blocks: self.hash_to_block_id.len(),
            block_size: self.block_size,
        }
    }
    
    /// Get a reference to a block
    pub fn get_block(&self, block_id: usize) -> Option<&Block> {
        self.blocks.get(block_id)
    }
    
    /// Get the block size
    pub fn block_size(&self) -> usize {
        self.block_size
    }
    
    /// Get the total number of blocks
    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }
}

/// Statistics about block manager usage
#[derive(Debug, Clone)]
pub struct BlockManagerStats {
    pub total_blocks: usize,
    pub free_blocks: usize,
    pub used_blocks: usize,
    pub cached_blocks: usize,
    pub block_size: usize,
}

impl BlockManagerStats {
    /// Calculate memory utilization percentage
    pub fn utilization(&self) -> f64 {
        if self.total_blocks == 0 {
            0.0
        } else {
            (self.used_blocks as f64 / self.total_blocks as f64) * 100.0
        }
    }
    
    /// Calculate cache hit ratio (approximate)
    pub fn cache_efficiency(&self) -> f64 {
        if self.used_blocks == 0 {
            0.0
        } else {
            (self.cached_blocks as f64 / self.used_blocks as f64) * 100.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::sampling_params::SamplingParams;
    
    #[test]
    fn test_block_creation() {
        let block = Block::new(42);
        assert_eq!(block.block_id, 42);
        assert_eq!(block.ref_count, 0);
        assert!(block.hash.is_none());
        assert!(block.token_ids.is_empty());
        assert!(block.is_free());
    }
    
    #[test]
    fn test_block_reference_counting() {
        let mut block = Block::new(0);
        
        block.add_ref();
        assert_eq!(block.ref_count, 1);
        assert!(!block.is_free());
        
        block.add_ref();
        assert_eq!(block.ref_count, 2);
        
        block.remove_ref();
        assert_eq!(block.ref_count, 1);
        
        block.remove_ref();
        assert_eq!(block.ref_count, 0);
        assert!(block.is_free());
    }
    
    #[test]
    fn test_block_manager_creation() {
        let bm = BlockManager::new(100, 256);
        assert_eq!(bm.num_blocks(), 100);
        assert_eq!(bm.block_size(), 256);
        assert_eq!(bm.free_block_ids.len(), 100);
        assert!(bm.used_block_ids.is_empty());
    }
    
    #[test]
    fn test_hash_computation() {
        let tokens1 = vec![1, 2, 3, 4, 5];
        let tokens2 = vec![1, 2, 3, 4, 5];
        let tokens3 = vec![1, 2, 3, 4, 6];
        
        let hash1 = BlockManager::compute_hash(&tokens1, None);
        let hash2 = BlockManager::compute_hash(&tokens2, None);
        let hash3 = BlockManager::compute_hash(&tokens3, None);
        
        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
        
        // Test with prefix
        let prefix_hash = Some(12345u64);
        let hash_with_prefix = BlockManager::compute_hash(&tokens1, prefix_hash);
        assert_ne!(hash1, hash_with_prefix);
    }
    
    #[test]
    fn test_sequence_allocation() {
        let mut bm = BlockManager::new(10, 4); // Small blocks for testing
        let prompt_tokens = vec![1, 2, 3, 4, 5, 6, 7, 8, 9]; // 9 tokens = 3 blocks
        let sampling_params = SamplingParams::new();
        let mut seq = Sequence::new(prompt_tokens, sampling_params);
        
        assert!(bm.can_allocate(&seq));
        assert!(bm.allocate(&mut seq).is_ok());
        
        assert_eq!(seq.block_table.len(), 3);
        assert_eq!(bm.free_block_ids.len(), 7);
        assert_eq!(bm.used_block_ids.len(), 3);
    }
    
    #[test]
    fn test_sequence_deallocation() {
        let mut bm = BlockManager::new(10, 4);
        let prompt_tokens = vec![1, 2, 3, 4, 5];
        let sampling_params = SamplingParams::new();
        let mut seq = Sequence::new(prompt_tokens, sampling_params);
        
        bm.allocate(&mut seq).unwrap();
        let initial_free = bm.free_block_ids.len();
        
        bm.deallocate(&mut seq);
        
        assert!(seq.block_table.is_empty());
        assert_eq!(seq.num_cached_tokens, 0);
        assert_eq!(bm.free_block_ids.len(), 10); // All blocks should be free again
        assert!(bm.used_block_ids.is_empty());
    }
    
    #[test]
    fn test_prefix_caching() {
        let mut bm = BlockManager::new(10, 4);
        let sampling_params = SamplingParams::new();
        
        // Create two sequences with shared prefix
        let tokens1 = vec![1, 2, 3, 4, 5, 6, 7, 8]; // 2 full blocks
        let tokens2 = vec![1, 2, 3, 4, 9, 10, 11, 12]; // Same first block, different second
        
        let mut seq1 = Sequence::new(tokens1, sampling_params.clone());
        let mut seq2 = Sequence::new(tokens2, sampling_params);
        
        // Allocate first sequence
        bm.allocate(&mut seq1).unwrap();
        let blocks_after_seq1 = bm.used_block_ids.len();
        
        // Allocate second sequence - should reuse first block
        bm.allocate(&mut seq2).unwrap();
        let blocks_after_seq2 = bm.used_block_ids.len();
        
        // Should have cached the first block
        assert!(seq2.num_cached_tokens > 0);
        
        // Verify reference counting
        let first_block_id = seq1.block_table[0] as usize;
        assert_eq!(bm.blocks[first_block_id].ref_count, 2);
    }
    
    #[test]
    fn test_append_operations() {
        let mut bm = BlockManager::new(10, 4);
        let prompt_tokens = vec![1, 2, 3]; // Partial block
        let sampling_params = SamplingParams::new();
        let mut seq = Sequence::new(prompt_tokens, sampling_params);
        
        bm.allocate(&mut seq).unwrap();
        assert_eq!(seq.block_table.len(), 1);
        
        // Add one more token to complete the block
        seq.append_token(4);
        assert!(bm.can_append(&seq));
        bm.may_append(&mut seq).unwrap();
        
        // Add another token to start a new block
        seq.append_token(5);
        assert!(bm.can_append(&seq));
        bm.may_append(&mut seq).unwrap();
        assert_eq!(seq.block_table.len(), 2);
    }
    
    #[test]
    fn test_block_manager_stats() {
        let mut bm = BlockManager::new(100, 256);
        let stats = bm.get_stats();
        
        assert_eq!(stats.total_blocks, 100);
        assert_eq!(stats.free_blocks, 100);
        assert_eq!(stats.used_blocks, 0);
        assert_eq!(stats.utilization(), 0.0);
        
        // Allocate some blocks
        let prompt_tokens = vec![1; 300]; // More than one block
        let sampling_params = SamplingParams::new();
        let mut seq = Sequence::new(prompt_tokens, sampling_params);
        
        bm.allocate(&mut seq).unwrap();
        let stats = bm.get_stats();
        
        assert!(stats.used_blocks > 0);
        assert!(stats.utilization() > 0.0);
    }
    
    #[test]
    fn test_insufficient_blocks() {
        let mut bm = BlockManager::new(2, 4); // Only 2 blocks
        let prompt_tokens = vec![1; 12]; // Needs 3 blocks
        let sampling_params = SamplingParams::new();
        let mut seq = Sequence::new(prompt_tokens, sampling_params);
        
        assert!(!bm.can_allocate(&seq));
        assert!(bm.allocate(&mut seq).is_err());
    }
}