# Building a RAG System with nano-vllm-rs

Implement a Retrieval-Augmented Generation (RAG) system that combines document retrieval with language model generation for accurate, up-to-date responses.

## 📋 Prerequisites

- Rust toolchain
- nano-vllm-rs
- `rust-bert` or `tch-rs` for embeddings
- `faiss-rs` or `hnsw-rs` for vector similarity search

## 🛠️ Implementation

### 1. Document Processing Pipeline

```rust
// src/rag/document_processor.rs
use rust_bert::pipelines::embedding::{Embedding, EmbeddingModel};
use std::path::Path;

pub struct DocumentProcessor {
    embedding_model: EmbeddingModel,
    chunk_size: usize,
    overlap: usize,
}

impl DocumentProcessor {
    pub fn new() -> Self {
        let embedding_model = EmbeddingModel::new(Default::default()).unwrap();
        Self {
            embedding_model,
            chunk_size: 512,
            overlap: 50,
        }
    }

    pub fn chunk_document(&self, text: &str) -> Vec<String> {
        // Implement text chunking with overlap
        // ...
        vec![]
    }

    
    pub fn embed_chunks(&self, chunks: &[String]) -> Vec<Vec<f32>> {
        self.embedding_model.embed(&chunks).unwrap()
    }
}
```

### 2. Vector Store

```rust
// src/rag/vector_store.rs
use hnsw_rs::prelude::*;
use std::collections::HashMap;

pub struct VectorStore {
    index: Hnsw<f32, DistCosine>,
    documents: HashMap<usize, String>,
    next_id: usize,
}

impl VectorStore {
    pub fn new(dim: usize) -> Self {
        let hnsw = Hnsw::new(16, dim, 16, DistCosine {}, 32, 200).unwrap();
        Self {
            index: hnsw,
            documents: HashMap::new(),
            next_id: 0,
        }
    }
    
    pub fn add_document(&mut self, text: String, embedding: Vec<f32>) {
        let id = self.next_id;
        self.documents.insert(id, text);
        self.index.add(&embedding, id).unwrap();
        self.next_id += 1;
    }
    
    pub fn search(&self, query_embedding: &[f32], k: usize) -> Vec<(usize, f32)> {
        let mut result = Vec::with_capacity(k);
        self.index.search(query_embedding, k, 24, &mut result);
        result
    }
}
```

### 3. RAG System

```rust
// src/rag/system.rs
use super::{document_processor::DocumentProcessor, vector_store::VectorStore};
use nano_vllm_rs::LLMEngine;

pub struct RAGSystem {
    processor: DocumentProcessor,
    vector_store: VectorStore,
    llm: LLMEngine,
}

impl RAGSystem {
    pub async fn new() -> Self {
        let processor = DocumentProcessor::new();
        let vector_store = VectorStore::new(768); // Dimension of embeddings
        let llm = LLMEngine::new(Default::default()).await.unwrap();
        
        Self {
            processor,
            vector_store,
            llm,
        }
    }
    
    pub fn add_document(&mut self, text: String) {
        let chunks = self.processor.chunk_document(&text);
        let embeddings = self.processor.embed_chunks(&chunks);
        
        for (chunk, embedding) in chunks.into_iter().zip(embeddings) {
            self.vector_store.add_document(chunk, embedding);
        }
    }
    
    pub async fn query(&self, question: &str, top_k: usize) -> String {
        // Get question embedding
        let query_embedding = self.processor.embed_chunks(&[question.to_string()])
            .pop()
            .unwrap();
        
        // Retrieve relevant chunks
        let results = self.vector_store.search(&query_embedding, top_k);
        let context: Vec<String> = results
            .into_iter()
            .filter_map(|(id, _)| self.vector_store.documents.get(&id).cloned())
            .collect();
        
        // Generate response using LLM
        let prompt = format!(
            "Context: {}\n\nQuestion: {}\n\nAnswer:",
            context.join("\n\n"),
            question
        );
        
        let response = self.llm
            .generate(vec![prompt], Default::default())
            .await
            .unwrap();
            
        response[0].text.clone()
    }
}
```

## 🚀 Usage Example

```rust
#[tokio::main]
async fn main() {
    // Initialize RAG system
    let mut rag = RAGSystem::new().await;
    
    // Add documents to the knowledge base
    let document = "Your document text here...";
    rag.add_document(document.to_string());
    
    // Query the system
    let question = "What is the main topic of the document?";
    let response = rag.query(question, 3).await;
    
    println!("Question: {}", question);
    println!("Response: {}", response);
}
```

## 📈 Advanced Techniques

### Hybrid Search
Combine vector search with keyword-based search for better results:

```rust
pub fn hybrid_search(&self, query: &str, k: usize) -> Vec<(usize, f32)> {
    // BM25 score (keyword-based)
    let bm25_scores = self.bm25_search(query, k);
    
    // Vector similarity score
    let query_embedding = self.processor.embed_chunks(&[query.to_string()])
        .pop()
        .unwrap();
    let vector_scores = self.vector_store.search(&query_embedding, k);
    
    // Combine scores (simple average)
    let mut combined = HashMap::new();
    
    for (id, score) in bm25_scores {
        *combined.entry(id).or_insert(0.0) += score * 0.5;
    }
    
    for (id, score) in vector_scores {
        *combined.entry(id).or_insert(0.0) += score * 0.5;
    }
    
    // Sort by combined score
    let mut results: Vec<_> = combined.into_iter().collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    results.into_iter().take(k).collect()
}
```

### Reranking
Improve retrieval quality with a cross-encoder reranker:

```rust
use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType,
};

pub struct Reranker {
    model: SentenceEmbeddingsModel,
}

impl Reranker {
    pub fn new() -> Self {
        let model = SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::AllMiniLmL12V2)
            .create_model()
            .unwrap();
        Self { model }
    }
    
    pub fn rerank(&self, query: &str, documents: &[String], top_k: usize) -> Vec<(usize, f32)> {
        let pairs: Vec<_> = documents
            .iter()
            .map(|doc| (query, doc.as_str()))
            .collect();
            
        let scores = self.model.compute_similarity(&pairs).unwrap();
        
        let mut results: Vec<_> = scores
            .into_iter()
            .enumerate()
            .map(|(i, score)| (i, score))
            .collect();
            
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.into_iter().take(top_k).collect()
    }
}
```

## 🧠 Best Practices

1. **Chunking Strategy**
   - Use semantic chunking (paragraphs, sections)
   - Consider overlap between chunks
   - Experiment with different chunk sizes

2. **Embedding Models**
   - Choose domain-appropriate models
   - Consider multilingual support
   - Evaluate different models on your data

3. **Retrieval Optimization**
   - Use HNSW or FAISS for fast approximate nearest neighbor search
   - Implement hybrid search for better recall
   - Add metadata filtering when applicable

4. **Prompt Engineering**
   - Clearly separate context from instructions
   - Include source attribution when possible
   - Handle cases where context is not found

## 📚 Additional Resources

- [Dense Passage Retrieval](https://arxiv.org/abs/2004.04906)
- [ColBERT: Efficient and Effective Passage Search](https://arxiv.org/abs/2004.12832)
- [LangChain RAG Implementation](https://python.langchain.com/docs/modules/chains/popular/vector_db_qa)

## 🛠️ Troubleshooting

### Common Issues

1. **Poor Retrieval Quality**
   - Try different embedding models
   - Adjust chunk size and overlap
   - Add more relevant documents

2. **Slow Performance**
   - Use approximate nearest neighbor search
   - Implement caching for frequent queries
   - Consider quantization of embeddings

3. **Hallucinations**
   - Add confidence thresholds
   - Implement fact-checking mechanisms
   - Use RAG with verification steps
