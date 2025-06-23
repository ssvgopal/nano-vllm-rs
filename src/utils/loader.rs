//! Model loading utilities
//! 
//! This module provides utilities for loading model weights from various formats,
//! particularly safetensors files used by HuggingFace models.

use std::collections::HashMap;
use std::path::Path;
use candle_core::{Tensor, Device};
use safetensors::SafeTensors;
use anyhow::{Result, Context as AnyhowContext};

/// Trait for loading weights into model parameters
pub trait WeightLoader {
    /// Load a weight tensor into this parameter
    fn load_weight(&mut self, weight: Tensor) -> Result<()>;
    
    /// Load a weight tensor with additional metadata (e.g., shard ID)
    fn load_weight_with_metadata(&mut self, weight: Tensor, metadata: &str) -> Result<()> {
        // Default implementation ignores metadata
        self.load_weight(weight)
    }
}

/// Default weight loader implementation for Tensor
impl WeightLoader for Tensor {
    fn load_weight(&mut self, weight: Tensor) -> Result<()> {
        // Verify shapes match
        if self.dims() != weight.dims() {
            anyhow::bail!(
                "Shape mismatch: expected {:?}, got {:?}",
                self.dims(),
                weight.dims()
            );
        }
        
        // Copy the weight data
        self.copy_(&weight)?;
        Ok(())
    }
}

/// Model loader for handling safetensors files
pub struct ModelLoader {
    /// Device to load tensors onto
    device: Device,
    
    /// Mapping from original parameter names to packed parameter names
    /// Format: original_name -> (packed_name, shard_id)
    packed_modules_mapping: HashMap<String, (String, String)>,
}

impl ModelLoader {
    /// Create a new model loader
    pub fn new(device: Device) -> Self {
        Self {
            device,
            packed_modules_mapping: HashMap::new(),
        }
    }
    
    /// Add a packed module mapping
    /// 
    /// This is used when multiple parameters are packed into a single tensor
    /// (e.g., q_proj, k_proj, v_proj packed into qkv_proj)
    pub fn add_packed_mapping<S1, S2, S3>(&mut self, original: S1, packed: S2, shard_id: S3)
    where
        S1: Into<String>,
        S2: Into<String>,
        S3: Into<String>,
    {
        self.packed_modules_mapping.insert(
            original.into(),
            (packed.into(), shard_id.into()),
        );
    }
    
    /// Load model weights from a safetensors file
    pub fn load_safetensors<P: AsRef<Path>>(
        &self,
        path: P,
        parameter_map: &mut HashMap<String, Box<dyn WeightLoader>>,
    ) -> Result<()> {
        let path = path.as_ref();
        let data = std::fs::read(path)
            .with_context(|| format!("Failed to read safetensors file: {:?}", path))?;
        
        let safetensors = SafeTensors::deserialize(&data)
            .with_context(|| format!("Failed to parse safetensors file: {:?}", path))?;
        
        for tensor_name in safetensors.names() {
            self.load_tensor(&safetensors, tensor_name, parameter_map)
                .with_context(|| format!("Failed to load tensor: {}", tensor_name))?;
        }
        
        Ok(())
    }
    
    /// Load all safetensors files in a directory
    pub fn load_safetensors_dir<P: AsRef<Path>>(
        &self,
        dir_path: P,
        parameter_map: &mut HashMap<String, Box<dyn WeightLoader>>,
    ) -> Result<()> {
        let dir_path = dir_path.as_ref();
        
        if !dir_path.is_dir() {
            anyhow::bail!("Path is not a directory: {:?}", dir_path);
        }
        
        let mut safetensors_files = Vec::new();
        
        for entry in std::fs::read_dir(dir_path)? {
            let entry = entry?;
            let path = entry.path();
            
            if path.extension().and_then(|s| s.to_str()) == Some("safetensors") {
                safetensors_files.push(path);
            }
        }
        
        if safetensors_files.is_empty() {
            anyhow::bail!("No safetensors files found in directory: {:?}", dir_path);
        }
        
        // Sort files for consistent loading order
        safetensors_files.sort();
        
        for file_path in safetensors_files {
            tracing::info!("Loading weights from: {:?}", file_path);
            self.load_safetensors(&file_path, parameter_map)?;
        }
        
        Ok(())
    }
    
    /// Load a single tensor from safetensors
    fn load_tensor(
        &self,
        safetensors: &SafeTensors,
        tensor_name: &str,
        parameter_map: &mut HashMap<String, Box<dyn WeightLoader>>,
    ) -> Result<()> {
        // Get tensor data from safetensors
        let tensor_view = safetensors.tensor(tensor_name)?;
        let tensor = Tensor::from_raw_buffer(
            tensor_view.data(),
            tensor_view.dtype().try_into()?,
            tensor_view.shape(),
            &self.device,
        )?;
        
        // Check if this tensor uses packed module mapping
        let (param_name, metadata) = if let Some((packed_name, shard_id)) = 
            self.find_packed_mapping(tensor_name) {
            (packed_name, Some(shard_id))
        } else {
            (tensor_name.to_string(), None)
        };
        
        // Find the parameter to load into
        if let Some(param_loader) = parameter_map.get_mut(&param_name) {
            if let Some(metadata) = metadata {
                param_loader.load_weight_with_metadata(tensor, &metadata)?;
            } else {
                param_loader.load_weight(tensor)?;
            }
        } else {
            tracing::warn!("No parameter found for tensor: {} (mapped to: {})", tensor_name, param_name);
        }
        
        Ok(())
    }
    
    /// Find packed module mapping for a tensor name
    fn find_packed_mapping(&self, tensor_name: &str) -> Option<(String, String)> {
        for (original_name, (packed_name, shard_id)) in &self.packed_modules_mapping {
            if tensor_name.contains(original_name) {
                let mapped_name = tensor_name.replace(original_name, packed_name);
                return Some((mapped_name, shard_id.clone()));
            }
        }
        None
    }
}

/// Helper function to create a standard model loader with common packed mappings
pub fn create_standard_loader(device: Device) -> ModelLoader {
    let mut loader = ModelLoader::new(device);
    
    // Common packed module mappings for transformer models
    loader.add_packed_mapping("q_proj", "qkv_proj", "q");
    loader.add_packed_mapping("k_proj", "qkv_proj", "k");
    loader.add_packed_mapping("v_proj", "qkv_proj", "v");
    loader.add_packed_mapping("gate_proj", "gate_up_proj", "0");
    loader.add_packed_mapping("up_proj", "gate_up_proj", "1");
    
    loader
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, DType};
    use tempfile::tempdir;
    use std::fs;
    
    struct MockWeightLoader {
        tensor: Tensor,
        loaded: bool,
        metadata: Option<String>,
    }
    
    impl MockWeightLoader {
        fn new(tensor: Tensor) -> Self {
            Self {
                tensor,
                loaded: false,
                metadata: None,
            }
        }
    }
    
    impl WeightLoader for MockWeightLoader {
        fn load_weight(&mut self, weight: Tensor) -> Result<()> {
            self.tensor.copy_(&weight)?;
            self.loaded = true;
            Ok(())
        }
        
        fn load_weight_with_metadata(&mut self, weight: Tensor, metadata: &str) -> Result<()> {
            self.load_weight(weight)?;
            self.metadata = Some(metadata.to_string());
            Ok(())
        }
    }
    
    #[test]
    fn test_model_loader_creation() {
        let device = Device::Cpu;
        let loader = ModelLoader::new(device);
        assert!(loader.packed_modules_mapping.is_empty());
    }
    
    #[test]
    fn test_packed_mapping() {
        let device = Device::Cpu;
        let mut loader = ModelLoader::new(device);
        
        loader.add_packed_mapping("q_proj", "qkv_proj", "q");
        loader.add_packed_mapping("k_proj", "qkv_proj", "k");
        
        assert_eq!(loader.packed_modules_mapping.len(), 2);
        
        let mapping = loader.find_packed_mapping("layer.0.self_attn.q_proj.weight");
        assert_eq!(mapping, Some(("layer.0.self_attn.qkv_proj.weight".to_string(), "q".to_string())));
    }
    
    #[test]
    fn test_standard_loader() {
        let device = Device::Cpu;
        let loader = create_standard_loader(device);
        
        // Should have common packed mappings
        assert!(!loader.packed_modules_mapping.is_empty());
        
        let q_mapping = loader.find_packed_mapping("q_proj");
        assert_eq!(q_mapping, Some(("qkv_proj".to_string(), "q".to_string())));
        
        let gate_mapping = loader.find_packed_mapping("gate_proj");
        assert_eq!(gate_mapping, Some(("gate_up_proj".to_string(), "0".to_string())));
    }
    
    #[test]
    fn test_weight_loader_trait() {
        let device = Device::Cpu;
        let mut tensor = Tensor::zeros((2, 3), DType::F32, &device).unwrap();
        let weight = Tensor::ones((2, 3), DType::F32, &device).unwrap();
        
        tensor.load_weight(weight).unwrap();
        
        let values: Vec<f32> = tensor.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(values, vec![1.0; 6]);
    }
    
    #[test]
    fn test_weight_loader_shape_mismatch() {
        let device = Device::Cpu;
        let mut tensor = Tensor::zeros((2, 3), DType::F32, &device).unwrap();
        let weight = Tensor::ones((3, 2), DType::F32, &device).unwrap(); // Wrong shape
        
        let result = tensor.load_weight(weight);
        assert!(result.is_err());
    }
}