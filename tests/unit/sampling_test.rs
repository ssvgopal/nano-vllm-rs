use nano_vllm_rs::{
    SamplingParams,
    sampling::{
        sample_top_k, sample_top_p, apply_temperature,
        apply_frequency_penalty, apply_presence_penalty
    }
};
use approx::assert_relative_eq;
use candle_core::{DType, Device, Tensor};

#[test]
fn test_sample_top_k() {
    let device = Device::Cpu;
    let logits = Tensor::new(&[1.0, 5.0, 2.0, 3.0, 4.0], &device).unwrap();
    
    // Test top-2 sampling
    let sampled = sample_top_k(&logits, 2).unwrap();
    let values = sampled.to_vec1::<f32>().unwrap();
    
    // Only top 2 values should remain, others should be -f32::INFINITY
    assert_eq!(values.len(), 5);
    assert!(values[1] > -f32::INFINITY);  // 5.0
    assert!(values[4] > -f32::INFINITY);  // 4.0
    
    // All other values should be -inf
    assert_eq!(values[0], -f32::INFINITY);
    assert_eq!(values[2], -f32::INFINITY);
    assert_eq!(values[3], -f32::INFINITY);
}

#[test]
fn test_sample_top_p() {
    let device = Device::Cpu;
    let logits = Tensor::new(&[1.0, 5.0, 2.0, 3.0, 4.0], &device).unwrap();
    
    // With p=0.9, it should keep top 2-3 values
    let sampled = sample_top_p(&logits, 0.9).unwrap();
    let values = sampled.to_vec1::<f32>().unwrap();
    
    // At least one value should be kept, at least one should be -inf
    assert!(values.iter().any(|&x| x > -f32::INFINITY));
    assert!(values.iter().any(|&x| x == -f32::INFINITY));
}

#[test]
fn test_apply_temperature() {
    let device = Device::Cpu;
    let logits = Tensor::new(&[1.0, 2.0], &device).unwrap();
    
    // With temperature = 0.5, differences should be amplified
    let result = apply_temperature(&logits, 0.5).unwrap();
    let values = result.to_vec1::<f32>().unwrap();
    
    // 1.0/0.5 = 2.0, 2.0/0.5 = 4.0
    assert_relative_eq!(values[0], 2.0, epsilon = 1e-5);
    assert_relative_eq!(values[1], 4.0, epsilon = 1e-5);
}

#[test]
fn test_frequency_penalty() {
    let device = Device::Cpu;
    let logits = Tensor::new(&[1.0, 2.0, 3.0], &device).unwrap();
    let freq = vec![1, 2];  // Tokens 1 and 2 have been seen
    
    // Penalty of 0.5 should reduce logits of seen tokens
    let result = apply_frequency_penalty(&logits, &freq, 0.5).unwrap();
    let values = result.to_vec1::<f32>().unwrap();
    
    // Token 0: no penalty (not in freq)
    assert_relative_eq!(values[0], 1.0, epsilon = 1e-5);
    // Token 1: 2.0 - (1 * 0.5) = 1.5
    assert_relative_eq!(values[1], 1.5, epsilon = 1e-5);
    // Token 2: 3.0 - (1 * 0.5) = 2.5
    assert_relative_eq!(values[2], 2.5, epsilon = 1e-5);
}

#[test]
fn test_presence_penalty() {
    let device = Device::Cpu;
    let logits = Tensor::new(&[1.0, 2.0, 3.0], &device).unwrap();
    let present = vec![1, 2];  // Tokens 1 and 2 are present
    
    // Penalty of 0.5 should reduce logits of present tokens
    let result = apply_presence_penalty(&logits, &present, 0.5).unwrap();
    let values = result.to_vec1::<f32>().unwrap();
    
    // Token 0: no penalty (not present)
    assert_relative_eq!(values[0], 1.0, epsilon = 1e-5);
    // Token 1: 2.0 - 0.5 = 1.5
    assert_relative_eq!(values[1], 1.5, epsilon = 1e-5);
    // Token 2: 3.0 - 0.5 = 2.5
    assert_relative_eq!(values[2], 2.5, epsilon = 1e-5);
}
