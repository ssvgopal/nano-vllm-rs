# Voice AI Applications with nano-vllm-rs

Build voice-enabled applications including voice assistants, call center automation, and real-time transcription using `nano-vllm-rs` with speech processing libraries.

## 📋 Prerequisites

- Rust toolchain
- `nano-vllm-rs`
- `cpal` for audio input/output
- `whisper-rs` for speech-to-text
- `tract` or `onnxruntime` for voice activity detection
- `rodio` for audio playback
- `deepgram` or `whisper.cpp` for cloud/offline STT

## 🛠️ Implementation

### 1. Audio Processing Module

```rust
// src/voice/audio_processor.rs
use anyhow::Result;
use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    Stream,
};
use std::sync::{Arc, Mutex};

pub struct AudioRecorder {
    sample_rate: u32,
    channels: u16,
    buffer: Arc<Mutex<Vec<f32>>>,
    stream: Option<Stream>,
}

impl AudioRecorder {
    pub fn new(sample_rate: u32, channels: u16) -> Self {
        Self {
            sample_rate,
            channels,
            buffer: Arc::new(Mutex::new(Vec::new())),
            stream: None,
        }
    }

    pub fn start_recording(&mut self) -> Result<()> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or_else(|| anyhow::anyhow!("No input device available"))?;

        let config = cpal::StreamConfig {
            channels: self.channels,
            sample_rate: cpal::SampleRate(self.sample_rate),
            buffer_size: cpal::BufferSize::Default,
        };

        let buffer = self.buffer.clone();
        let err_fn = |err| eprintln!("Error in audio stream: {}", err);

        let stream = device.build_input_stream(
            &config,
            move |data: &[f32], _: &_| {
                let mut buffer = buffer.lock().unwrap();
                buffer.extend_from_slice(data);
            },
            err_fn,
            None,
        )?;

        stream.play()?;
        self.stream = Some(stream);
        Ok(())
    }

    pub fn stop_recording(&mut self) -> Vec<f32> {
        if let Some(stream) = self.stream.take() {
            let _ = stream.pause();
        }
        let buffer = self.buffer.lock().unwrap();
        buffer.clone()
    }
}
```

### 2. Speech-to-Text with Whisper

```rust
// src/voice/speech_to_text.rs
use anyhow::Result;
use whisper_rs::{WhisperContext, FullParams, SamplingStrategy};

pub struct SpeechToText {
    ctx: WhisperContext,
}

impl SpeechToText {
    pub fn new(model_path: &str) -> Result<Self> {
        let ctx = WhisperContext::new(model_path)
            .map_err(|e| anyhow::anyhow!("Failed to load model: {}", e))?;
        Ok(Self { ctx })
    }

    pub fn transcribe(&self, audio_data: &[f32], language: Option<&str>) -> Result<String> {
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        
        if let Some(lang) = language {
            params = params.language(Some(lang));
        }
        
        let state = self.ctx.create_state()?;
        state.full(params, audio_data)?;
        
        let num_segments = state.full_n_segments()?;
        let mut transcription = String::new();
        
        for i in 0..num_segments {
            let segment = state.full_get_segment_text(i)?;
            transcription.push_str(&segment);
            transcription.push(' ');
        }
        
        Ok(transcription.trim().to_string())
    }
}
```

### 3. Text-to-Speech with Piper

```rust
// src/voice/text_to_speech.rs
use anyhow::Result;
use piper_rs::{Piper, Voice};
use rodio::Sink;
use std::io::Cursor;

pub struct TextToSpeech {
    piper: Piper,
    voice: Voice,
}

impl TextToSpeech {
    pub async fn new(model_path: &str, config_path: &str) -> Result<Self> {
        let piper = Piper::new()?;
        let voice = Voice::from_files(model_path, config_path).await?;
        Ok(Self { piper, voice })
    }
    
    pub fn speak(&self, text: &str) -> Result<()> {
        let (_stream, stream_handle) = rodio::OutputStream::try_default()?;
        let sink = Sink::try_new(&stream_handle)?;
        
        let audio_data = self.piper.synthesize(text, &self.voice)?;
        let cursor = Cursor::new(audio_data);
        
        sink.append(rodio::Decoder::new(cursor)?);
        sink.sleep_until_end();
        
        Ok(())
    }
}
```

### 4. Voice Activity Detection

```rust
// src/voice/vad.rs
use anyhow::Result;
use tract_onnx::prelude::*;

pub struct VoiceActivityDetector {
    model: TypedRunnableModel<TypedModel>,
    threshold: f32,
}

impl VoiceActivityDetector {
    pub async fn new(model_path: &str, threshold: f32) -> Result<Self> {
        let model = tract_onnx::onnx()
            .model_for_path(model_path)?
            .with_input_fact(0, f32::fact([1, 1, 80, 300]).into())?
            .into_typed()?;
            
        Ok(Self { model, threshold })
    }
    
    pub fn detect_voice_activity(&self, audio_chunk: &[f32]) -> Result<bool> {
        // Preprocess audio (MFCCs, normalization, etc.)
        let features = self.extract_features(audio_chunk)?;
        
        // Run inference
        let result = self.model.run(tvec!(features.into()))?;
        let output = result[0].to_array_view::<f32>()?;
        
        // Simple threshold-based VAD
        let speech_prob = output[1]; // Assuming binary classification
        Ok(speech_prob > self.threshold)
    }
    
    fn extract_features(&self, audio: &[f32]) -> Result<Tensor> {
        // Implement feature extraction (MFCCs, etc.)
        // This is a simplified example
        let mut features = vec![0.0; 80 * 300];
        // ... feature extraction logic ...
        
        Tensor::from_shape(&[1, 1, 80, 300], &features)
            .map_err(Into::into)
    }
}
```

## 🚀 Usage Examples

### 1. Voice Assistant

```rust
#[tokio::main]
async fn main() -> Result<()> {
    // Initialize components
    let stt = SpeechToText::new("models/whisper-tiny.en")?;
    let tts = TextToSpeech::new("models/piper/en/en_US-lessac-medium").await?;
    let vad = VoiceActivityDetector::new("models/silero_vad.onnx", 0.5).await?;
    let mut recorder = AudioRecorder::new(16000, 1);
    
    // Initialize LLM
    let llm = LLMEngine::new(Default::default()).await?;
    
    println!("Listening... (say 'exit' to quit)");
    
    loop {
        // Record until voice activity stops
        recorder.start_recording()?;
        
        // Simple VAD loop
        let mut silent_frames = 0;
        let mut audio_buffer = Vec::new();
        
        while silent_frames < 5 { // Wait for 5 silent frames
            std::thread::sleep(std::time::Duration::from_millis(100));
            let chunk = recorder.stop_recording();
            
            if vad.detect_voice_activity(&chunk)? {
                audio_buffer.extend(chunk);
                silent_frames = 0;
            } else {
                silent_frames += 1;
            }
            
            recorder.start_recording()?;
        }
        
        // Transcribe audio
        let text = stt.transcribe(&audio_buffer, Some("en"))?;
        
        if text.to_lowercase().contains("exit") {
            break;
        }
        
        println!("You: {}", text);
        
        // Generate response
        let response = llm.generate(
            vec![format!("User: {}\nAssistant:", text)],
            Default::default()
        ).await?;
        
        println!("Assistant: {}", response[0].text);
        
        // Speak response
        tts.speak(&response[0].text)?;
    }
    
    Ok(())
}
```

### 2. Real-Time Transcription

```rust
pub struct RealTimeTranscriber {
    stt: SpeechToText,
    vad: VoiceActivityDetector,
    sample_rate: u32,
    buffer: Vec<f32>,
}

impl RealTimeTranscriber {
    pub fn new() -> Result<Self> {
        Ok(Self {
            stt: SpeechToText::new("models/whisper-tiny.en")?,
            vad: VoiceActivityDetector::new("models/silero_vad.onnx", 0.5).await?,
            sample_rate: 16000,
            buffer: Vec::with_capacity(16000 * 10), // 10 seconds buffer
        })
    }
    
    pub async fn process_chunk(&mut self, audio_chunk: &[f32]) -> Result<Option<String>> {
        self.buffer.extend(audio_chunk);
        
        // Process in chunks of 3 seconds
        let chunk_size = self.sample_rate as usize * 3;
        
        if self.buffer.len() >= chunk_size {
            let chunk: Vec<f32> = self.buffer.drain(..chunk_size).collect();
            
            if self.vad.detect_voice_activity(&chunk)? {
                let text = self.stt.transcribe(&chunk, Some("en"))?;
                return Ok(Some(text));
            }
        }
        
        Ok(None)
    }
}
```

## 📈 Advanced Techniques

### 1. Speaker Diarization

```rust
pub struct SpeakerDiarizer {
    model: tract_onnx::prelude::TypedRunnableModel<tract_onnx::prelude::TypedModel>,
    sample_rate: u32,
}

impl SpeakerDiarizer {
    pub async fn new(model_path: &str) -> Result<Self> {
        let model = tract_onnx::onnx()
            .model_for_path(model_path)?
            .with_input_fact(0, f32::fact([1, 80, 300]).into())?
            .into_typed()?;
            
        Ok(Self {
            model,
            sample_rate: 16000,
        })
    }
    
    pub fn diarize(&self, audio: &[f32]) -> Result<Vec<SpeakerSegment>> {
        let features = self.extract_features(audio)?;
        let output = self.model.run(tvec!(features.into()))?;
        
        // Process output to get speaker segments
        // This is a simplified example
        let segments = self.process_output(output)?;
        
        Ok(segments)
    }
    
    // ... implementation details ...
}
```

### 2. Emotion Recognition from Voice

```rust
pub struct EmotionRecognizer {
    model: tract_onnx::prelude::TypedRunnableModel<tract_onnx::prelude::TypedModel>,
    classes: Vec<&'static str>,
}

impl EmotionRecognizer {
    pub async fn new(model_path: &str) -> Result<Self> {
        let model = tract_onnx::onnx()
            .model_for_path(model_path)?
            .with_input_fact(0, f32::fact([1, 1, 80, 300]).into())?
            .into_typed()?;
            
        let classes = vec!["neutral", "happy", "sad", "angry", "fear", "disgust", "surprise"];
        
        Ok(Self { model, classes })
    }
    
    pub fn recognize_emotion(&self, audio: &[f32]) -> Result<&str> {
        let features = self.extract_features(audio)?;
        let output = self.model.run(tvec!(features.into()))?;
        
        // Get class with highest probability
        let probs = output[0].to_array_view::<f32>()?;
        let max_idx = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);
            
        Ok(self.classes[max_idx])
    }
    
    // ... implementation details ...
}
```

## 🧠 Best Practices

1. **Audio Quality**
   - Use appropriate sample rates (16kHz is common for speech)
   - Implement noise reduction
   - Consider acoustic echo cancellation

2. **Performance**
   - Use streaming inference for real-time applications
   - Implement batching for better throughput
   - Consider model quantization for edge deployment

3. **User Experience**
   - Provide visual feedback during recording
   - Implement wake word detection
   - Handle background noise gracefully

4. **Privacy**
   - Process audio locally when possible
   - Implement data anonymization
   - Provide clear privacy controls

## 📚 Additional Resources

- [Mozilla's DeepSpeech](https://github.com/mozilla/DeepSpeech)
- [Coqui TTS](https://github.com/coqui-ai/TTS)
- [Silero VAD](https://github.com/snakers4/silero-vad)
- [NVIDIA Riva](https://developer.nvidia.com/riva)

## 🛠️ Troubleshooting

### Common Issues

1. **Poor Transcription Quality**
   - Check audio input levels
   - Try different models (larger models may be more accurate)
   - Consider adding language model for post-processing

2. **High Latency**
   - Reduce model size
   - Use streaming inference
   - Optimize feature extraction

3. **Background Noise**
   - Implement noise suppression
   - Use a better microphone
   - Train with noisy data

4. **Memory Usage**
   - Use smaller models
   - Enable model quantization
   - Implement model unloading when not in use
