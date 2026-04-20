mod pool;

// `onnx` links pyke's prebuilt ONNX Runtime at build time; `onnx-system-ort`
// dlopen()s an already-installed libonnxruntime at runtime. Enabling both
// pulls in conflicting `ort` configurations, so reject the combination up
// front with a clear error instead of a confusing link failure.
#[cfg(all(feature = "onnx", feature = "onnx-system-ort"))]
compile_error!(
    "features `onnx` and `onnx-system-ort` are mutually exclusive — enable exactly one."
);

#[cfg(any(feature = "onnx", feature = "onnx-system-ort"))]
mod onnx;

use plume_core::error::PlumeError;
use plume_core::types::{MultiVector, EMBEDDING_DIM};

pub use pool::pool_vectors;

/// Trait for encoding text into multi-vector (ColBERT) representations.
pub trait Encode: Send + Sync {
    fn encode_single(&self, text: &str) -> Result<MultiVector, PlumeError>;
    fn encode_batch(&self, texts: &[&str]) -> Result<Vec<MultiVector>, PlumeError>;
    /// Human-readable identifier for observability: `mock`, or
    /// `onnx:{model-id}`. Surfaced via `/health` so clients can warn
    /// users that the mock encoder produces meaningless embeddings.
    fn kind(&self) -> String {
        "unknown".into()
    }
}

/// Build the best available encoder for the given config.
///
/// If the `onnx` feature is enabled and the model directory exists with
/// `model.onnx` + `tokenizer.json`, uses the real ONNX encoder.
/// Otherwise falls back to MockEncoder.
pub fn build_encoder(config: &plume_core::config::EncoderConfig) -> Box<dyn Encode> {
    #[cfg(any(feature = "onnx", feature = "onnx-system-ort"))]
    {
        let model_path = std::path::Path::new(&config.model);
        if model_path.join("model.onnx").exists() && model_path.join("tokenizer.json").exists() {
            match onnx::Encoder::with_config(model_path, config) {
                Ok(enc) => {
                    tracing::info!(model = %config.model, "using ONNX encoder");
                    return Box::new(enc);
                }
                Err(e) => {
                    tracing::warn!(error = %e, "failed to load ONNX encoder, falling back to mock");
                }
            }
        }
    }

    tracing::info!("using mock encoder (pool_factor={})", config.pool_factor);
    Box::new(MockEncoder::new(config.pool_factor))
}

// --- MockEncoder ---

/// A mock encoder for development/testing without ONNX models.
///
/// Produces deterministic pseudo-embeddings from text hashes.
pub struct MockEncoder {
    dim: usize,
    tokens_per_doc: usize,
    pool_factor: usize,
}

impl MockEncoder {
    pub fn new(pool_factor: usize) -> Self {
        Self {
            dim: EMBEDDING_DIM,
            tokens_per_doc: 64,
            pool_factor,
        }
    }
}

impl Encode for MockEncoder {
    fn kind(&self) -> String {
        "mock".into()
    }

    fn encode_single(&self, text: &str) -> Result<MultiVector, PlumeError> {
        Ok(self.encode_batch(&[text])?.into_iter().next().unwrap())
    }

    fn encode_batch(&self, texts: &[&str]) -> Result<Vec<MultiVector>, PlumeError> {
        Ok(texts
            .iter()
            .map(|text| {
                let hash = simple_hash(text);
                let n_tokens = (self.tokens_per_doc / self.pool_factor).max(1);

                (0..n_tokens)
                    .map(|t| {
                        let mut vec: Vec<f32> = (0..self.dim)
                            .map(|d| {
                                ((hash.wrapping_add(t as u64 * 31).wrapping_add(d as u64 * 7))
                                    % 1000) as f32
                                    / 1000.0
                                    - 0.5
                            })
                            .collect();
                        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                        if norm > 1e-12 {
                            vec.iter_mut().for_each(|x| *x /= norm);
                        }
                        vec
                    })
                    .collect()
            })
            .collect())
    }
}

// --- ONNX Encoder Encode impl ---

#[cfg(any(feature = "onnx", feature = "onnx-system-ort"))]
impl Encode for onnx::Encoder {
    fn kind(&self) -> String {
        format!("onnx:{}", self.model_id())
    }

    fn encode_single(&self, text: &str) -> Result<MultiVector, PlumeError> {
        self.encode_single(text)
    }

    fn encode_batch(&self, texts: &[&str]) -> Result<Vec<MultiVector>, PlumeError> {
        self.encode_batch(texts)
    }
}

fn simple_hash(s: &str) -> u64 {
    let mut h: u64 = 5381;
    for b in s.bytes() {
        h = h.wrapping_mul(33).wrapping_add(b as u64);
    }
    h
}
