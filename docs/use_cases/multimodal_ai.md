# Multimodal AI with Image and Text Understanding

Build applications that can process and understand both images and text using `nano-vllm-rs` with vision models.

## 📋 Prerequisites

- Rust toolchain
- `nano-vllm-rs` with vision support
- `tch-rs` for tensor operations
- `image` crate for image processing
- CUDA (recommended for better performance)

## 🛠️ Implementation

### 1. Image Processing Module

```rust
// src/multimodal/image_processor.rs
use anyhow::Result;
use tch::{Device, Kind, Tensor, vision::image as tch_image};
use image::{DynamicImage, ImageBuffer};

pub struct ImageProcessor {
    image_size: (i64, i64),
    mean: [f64; 3],
    std: [f64; 3],
}

impl ImageProcessor {
    pub fn new(image_size: (i64, i64)) -> Self {
        Self {
            image_size,
            mean: [0.485, 0.456, 0.406],  // ImageNet mean
            std: [0.229, 0.224, 0.225],    // ImageNet std
        }
    }
    
    pub fn load_and_preprocess(&self, image_path: &str) -> Result<Tensor> {
        // Load image
        let image = tch_image::load(image_path)?
            .resize(self.image_size.0, self.image_size.1)?;
        
        // Convert to tensor and normalize
        let mut tensor = image.to_kind(Kind::Float) / 255.0;
        
        // Normalize with ImageNet stats
        for (i, (&m, &s)) in self.mean.iter().zip(&self.std).enumerate() {
            let channel = tensor.get(i);
            tensor.get_mut(i).copy_(&((channel - m) / s));
        }
        
        // Add batch dimension
        Ok(tensor.unsqueeze(0))
    }
    
    pub fn tensor_to_image(&self, tensor: &Tensor) -> DynamicImage {
        let (_, c, h, w) = tensor.size4().unwrap();
        let mut tensor = tensor.squeeze().detach();
        
        // Denormalize
        for (i, (&m, &s)) in self.mean.iter().zip(&self.std).enumerate() {
            let channel = tensor.get(i);
            tensor.get_mut(i).copy_(&(channel * s + m));
        }
        
        // Convert to u8
        let tensor = (tensor * 255.0).clamp(0.0, 255.0).to_kind(Kind::Uint8);
        
        // Convert to image
        let image_data = Vec::<u8>::try_from(&tensor).unwrap();
        let img_buf = ImageBuffer::from_raw(w as u32, h as u32, image_data).unwrap();
        DynamicImage::ImageRgb8(img_buf)
    }
}
```

### 2. Vision-Language Model Wrapper

```rust
// src/multimodal/vision_model.rs
use anyhow::Result;
use tch::{Device, Tensor};
use serde_json::Value;

pub struct VisionLanguageModel {
    model: Box<dyn VisionLanguageModelTrait>,
    device: Device,
}

#[async_trait::async_trait]
pub trait VisionLanguageModelTrait: Send + Sync {
    fn embed_image(&self, image: &Tensor) -> Result<Tensor>;
    fn embed_text(&self, text: &str) -> Result<Tensor>;
    async fn generate_from_image(&self, image: &Tensor, prompt: &str) -> Result<String>;
    async fn answer_question(&self, image: &Tensor, question: &str) -> Result<String>;
}

impl VisionLanguageModel {
    pub fn new(model_name: &str, device: Device) -> Result<Self> {
        // This would initialize the specific model implementation
        // For example: CLIP, LLaVA, etc.
        let model: Box<dyn VisionLanguageModelTrait> = match model_name {
            "clip" => Box::new(ClipModel::new(device)?),
            "llava" => Box::new(LlavaModel::new(device)?),
            _ => return Err(anyhow::anyhow!("Unsupported model")),
        };
        
        Ok(Self { model, device })
    }
    
    pub async fn get_image_caption(&self, image_tensor: &Tensor) -> Result<String> {
        self.model.generate_from_image(image_tensor, "A photo of").await
    }
    
    pub async fn get_image_embeddings(&self, image_tensor: &Tensor) -> Result<Tensor> {
        self.model.embed_image(image_tensor)
    }
    
    pub async fn get_text_embeddings(&self, text: &str) -> Result<Tensor> {
        self.model.embed_text(text)
    }
    
    pub async fn answer_question_about_image(
        &self, 
        image_tensor: &Tensor, 
        question: &str
    ) -> Result<String> {
        self.model.answer_question(image_tensor, question).await
    }
    
    pub async fn find_similar_images(
        &self,
        query_embedding: &Tensor,
        image_embeddings: &[Tensor],
        top_k: usize,
    ) -> Result<Vec<(usize, f32)>> {
        // Calculate cosine similarity
        let query_norm = query_embedding.f_norm_scalaropt(1, false, Kind::Float)?;
        let mut results = Vec::new();
        
        for (i, emb) in image_embeddings.iter().enumerate() {
            let dot_product = query_embedding.dot(emb)?;
            let emb_norm = emb.f_norm_scalaropt(1, false, Kind::Float)?;
            let similarity = dot_product / (query_norm * emb_norm + 1e-8);
            results.push((i, similarity as f32));
        }
        
        // Sort by similarity (descending)
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Return top-k results
        Ok(results.into_iter().take(top_k).collect())
    }
}
```

### 3. Example Implementation: CLIP Model

```rust
// src/multimodal/models/clip.rs
use anyhow::Result;
use tch::{Device, Tensor, Kind};
use std::path::Path;

pub struct ClipModel {
    model: clip_rs::CLIP,
    device: Device,
}

impl ClipModel {
    pub fn new(device: Device) -> Result<Self> {
        let model = clip_rs::CLIP::from_pretrained("openai/clip-vit-base-patch32")?;
        Ok(Self { model, device })
    }
}

#[async_trait::async_trait]
impl VisionLanguageModelTrait for ClipModel {
    fn embed_image(&self, image: &Tensor) -> Result<Tensor> {
        let features = self.model.forward_image(image.to_device(self.device))?;
        Ok(features)
    }
    
    fn embed_text(&self, text: &str) -> Result<Tensor> {
        let text_tensor = self.model.tokenize(&[text])?;
        let features = self.model.forward_text(&text_tensor.to_device(self.device))?;
        Ok(features)
    }
    
    async fn generate_from_image(&self, _image: &Tensor, _prompt: &str) -> Result<String> {
        // CLIP doesn't support generation, this is a placeholder
        Err(anyhow::anyhow!("CLIP doesn't support text generation"))
    }
    
    async fn answer_question(&self, _image: &Tensor, _question: &str) -> Result<String> {
        // CLIP doesn't support QA, this is a placeholder
        Err(anyhow::anyhow!("CLIP doesn't support question answering"))
    }
}
```

## 🚀 Usage Example

```rust
#[tokio::main]
async fn main() -> Result<()> {
    // Initialize components
    let device = Device::cuda_if_available();
    let image_processor = ImageProcessor::new((224, 224));
    let model = VisionLanguageModel::new("clip", device)?;
    
    // Process image
    let image_tensor = image_processor.load_and_preprocess("image.jpg")?;
    
    // Get image caption
    let caption = model.get_image_caption(&image_tensor).await?;
    println!("Image caption: {}", caption);
    
    // Get image embeddings
    let image_embedding = model.get_image_embeddings(&image_tensor).await?;
    
    // Compare with text
    let text_embedding = model.get_text_embeddings("a photo of a cat").await?;
    
    // Calculate similarity
    let similarity = image_embedding.cosine_similarity(&text_embedding, 0, 1e-8)?;
    println!("Similarity with 'a photo of a cat': {:.2}%", similarity * 100.0);
    
    // For models that support it:
    let answer = model.answer_question_about_image(
        &image_tensor, 
        "What is the main subject of this image?"
    ).await?;
    println!("Q: What is the main subject of this image?\nA: {}", answer);
    
    Ok(())
}
```

## 📈 Advanced Techniques

### Multimodal RAG (Retrieval Augmented Generation)

```rust
pub struct MultimodalRAG {
    vision_model: Arc<VisionLanguageModel>,
    text_engine: LLMEngine,
    vector_store: VectorStore,
}

impl MultimodalRAG {
    pub async fn new() -> Result<Self> {
        let device = Device::cuda_if_available();
        let vision_model = Arc::new(VisionLanguageModel::new("clip", device)?);
        let text_engine = LLMEngine::new(Default::default()).await?;
        let vector_store = VectorStore::new(512); // CLIP embedding size
        
        Ok(Self {
            vision_model,
            text_engine,
            vector_store,
        })
    }
    
    pub async fn add_document(&self, text: &str, image_path: Option<&str>) -> Result<()> {
        let mut embeddings = Vec::new();
        
        // Get text embedding
        if !text.is_empty() {
            let text_emb = self.vision_model.embed_text(text).await?;
            embeddings.push(text_emb);
        }
        
        // Get image embedding if available
        if let Some(path) = image_path {
            let image_processor = ImageProcessor::new((224, 224));
            let image_tensor = image_processor.load_and_preprocess(path)?;
            let image_emb = self.vision_model.embed_image(&image_tensor).await?;
            embeddings.push(image_emb);
        }
        
        // Average embeddings if both text and image
        let combined_emb = if embeddings.len() > 1 {
            let sum: Tensor = embeddings.iter().sum();
            sum / (embeddings.len() as f64)
        } else {
            embeddings.into_iter().next().unwrap()
        };
        
        // Store in vector DB
        self.vector_store.add_embedding(combined_emb, text);
        
        Ok(())
    }
    
    pub async fn query(&self, query: &str, image_path: Option<&str>) -> Result<String> {
        // Get query embedding (text + image if available)
        let mut query_emb = self.vision_model.embed_text(query).await?;
        
        if let Some(path) = image_path {
            let image_processor = ImageProcessor::new((224, 224));
            let image_tensor = image_processor.load_and_preprocess(path)?;
            let image_emb = self.vision_model.embed_image(&image_tensor).await?;
            
            // Average with text embedding
            query_emb = (query_emb + image_emb)? / 2.0;
        }
        
        // Retrieve relevant documents
        let results = self.vector_store.search(&query_emb, 3);
        
        // Format context
        let context: String = results
            .into_iter()
            .map(|(text, _)| text)
            .collect::<Vec<_>>()
            .join("\n\n");
        
        // Generate response
        let prompt = format!(
            "Context:\n{}\n\nQuestion: {}\n\nAnswer:",
            context, query
        );
        
        let response = self.text_engine
            .generate(vec![prompt], Default::default())
            .await?;
            
        Ok(response[0].text.clone())
    }
}
```

## 🧠 Best Practices

1. **Model Selection**
   - Choose models based on your specific needs (CLIP for retrieval, LLaVA for generation, etc.)
   - Consider model size vs. accuracy trade-offs
   - Use quantized models for edge deployment

2. **Preprocessing**
   - Normalize images according to model requirements
   - Handle different aspect ratios appropriately
   - Use data augmentation for training

3. **Efficiency**
   - Cache embeddings for static content
   - Use batching for processing multiple images
   - Consider using ONNX Runtime for optimized inference

4. **Evaluation**
   - Use appropriate metrics (CLIPScore, CIDEr, etc.)
   - Perform human evaluation for qualitative assessment
   - Monitor for bias and fairness

## 📚 Additional Resources

- [CLIP: Connecting Text and Images](https://openai.com/research/clip)
- [LLaVA: Large Language and Vision Assistant](https://llava-vl.github.io/)
- [Multimodal Machine Learning: A Survey and Taxonomy](https://arxiv.org/abs/1705.09406)

## 🛠️ Troubleshooting

### Common Issues

1. **Poor Image Understanding**
   - Try different vision backbones
   - Increase image resolution if possible
   - Use domain-specific fine-tuning

2. **High Memory Usage**
   - Reduce batch size
   - Use gradient checkpointing
   - Enable mixed precision training

3. **Slow Inference**
   - Use model quantization
   - Enable TensorRT optimization
   - Use a smaller model variant

4. **Alignment Issues**
   - Check preprocessing steps
   - Ensure consistent normalization
   - Verify model input requirements
