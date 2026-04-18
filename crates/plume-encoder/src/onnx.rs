use std::path::Path;
use std::sync::Mutex;

use ndarray::Array2;
use ort::session::Session;
use ort::value::Tensor;
use plume_core::config::EncoderConfig;
use plume_core::error::PlumeError;
use plume_core::types::MultiVector;
use tokenizers::Tokenizer;
use tracing::info;

use crate::pool::pool_vectors;

/// ColBERT-style encoder that produces multi-vector (token-level) embeddings.
///
/// Wraps an ONNX model (e.g. LateOn-Code-edge) and HuggingFace tokenizer.
/// Session is behind a Mutex because ort 2.0 requires `&mut self` for `run()`.
pub struct Encoder {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    pool_factor: usize,
}

impl Encoder {
    /// Create a new encoder from a local ONNX model directory.
    ///
    /// The directory should contain:
    /// - `model.onnx` — the ONNX model file
    /// - `tokenizer.json` — HuggingFace tokenizer
    pub fn from_directory(model_dir: &Path) -> Result<Self, PlumeError> {
        let model_path = model_dir.join("model.onnx");
        let tokenizer_path = model_dir.join("tokenizer.json");

        if !model_path.exists() {
            return Err(PlumeError::Encoder(format!(
                "model.onnx not found in {}",
                model_dir.display()
            )));
        }
        if !tokenizer_path.exists() {
            return Err(PlumeError::Encoder(format!(
                "tokenizer.json not found in {}",
                model_dir.display()
            )));
        }

        let session = Session::builder()
            .map_err(|e| PlumeError::Encoder(format!("failed to create session builder: {e}")))?
            .with_intra_threads(4)
            .map_err(|e| PlumeError::Encoder(format!("failed to set threads: {e}")))?
            .commit_from_file(&model_path)
            .map_err(|e| PlumeError::Encoder(format!("failed to load model: {e}")))?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| PlumeError::Encoder(format!("failed to load tokenizer: {e}")))?;

        info!("encoder loaded from {}", model_dir.display());

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            pool_factor: 2,
        })
    }

    /// Create encoder with custom config.
    pub fn with_config(model_dir: &Path, config: &EncoderConfig) -> Result<Self, PlumeError> {
        let mut encoder = Self::from_directory(model_dir)?;
        encoder.pool_factor = config.pool_factor;
        Ok(encoder)
    }

    /// Encode a single text into a multi-vector representation.
    pub fn encode_single(&self, text: &str) -> Result<MultiVector, PlumeError> {
        let results = self.encode_batch(&[text])?;
        Ok(results.into_iter().next().unwrap())
    }

    /// Encode a batch of texts into multi-vector representations.
    pub fn encode_batch(&self, texts: &[&str]) -> Result<Vec<MultiVector>, PlumeError> {
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| PlumeError::Encoder(format!("tokenization failed: {e}")))?;

        let batch_size = encodings.len();
        let max_len = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0);

        // Build input tensors: input_ids and attention_mask
        let mut input_ids = vec![0i64; batch_size * max_len];
        let mut attention_mask = vec![0i64; batch_size * max_len];

        for (i, encoding) in encodings.iter().enumerate() {
            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();
            for (j, (&id, &m)) in ids.iter().zip(mask.iter()).enumerate() {
                input_ids[i * max_len + j] = id as i64;
                attention_mask[i * max_len + j] = m as i64;
            }
        }

        let input_ids_array = ndarray::Array2::from_shape_vec((batch_size, max_len), input_ids)
            .map_err(|e| PlumeError::Encoder(format!("shape error: {e}")))?;

        let attention_mask_array =
            ndarray::Array2::from_shape_vec((batch_size, max_len), attention_mask)
                .map_err(|e| PlumeError::Encoder(format!("shape error: {e}")))?;

        let input_ids_tensor = Tensor::from_array(input_ids_array)
            .map_err(|e| PlumeError::Encoder(format!("tensor creation failed: {e}")))?;
        let attention_mask_tensor = Tensor::from_array(attention_mask_array)
            .map_err(|e| PlumeError::Encoder(format!("tensor creation failed: {e}")))?;

        let mut session = self
            .session
            .lock()
            .map_err(|e| PlumeError::Encoder(format!("session lock poisoned: {e}")))?;

        let outputs = session
            .run(ort::inputs![input_ids_tensor, attention_mask_tensor])
            .map_err(|e| PlumeError::Encoder(format!("inference failed: {e}")))?;

        // Output shape: [batch_size, seq_len, embedding_dim]
        let output_view = outputs[0]
            .try_extract_array::<f32>()
            .map_err(|e| PlumeError::Encoder(format!("output extraction failed: {e}")))?;

        let shape = output_view.shape();
        let embedding_dim = shape[2];

        let output_array = output_view
            .to_shape((batch_size, max_len, embedding_dim))
            .map_err(|e| PlumeError::Encoder(format!("reshape failed: {e}")))?;

        let mut results = Vec::with_capacity(batch_size);

        for (i, encoding) in encodings.iter().enumerate() {
            let seq_len = encoding
                .get_attention_mask()
                .iter()
                .filter(|&&m| m == 1)
                .count();
            let token_embeddings: Array2<f32> =
                output_array.slice(ndarray::s![i, ..seq_len, ..]).to_owned();

            let normalized = l2_normalize_rows(&token_embeddings);

            let pooled = if self.pool_factor > 1 {
                pool_vectors(&normalized, self.pool_factor)
            } else {
                normalized
            };

            let mv: MultiVector = pooled.outer_iter().map(|row| row.to_vec()).collect();

            results.push(mv);
        }

        Ok(results)
    }
}

/// L2-normalize each row of a 2D array.
fn l2_normalize_rows(arr: &Array2<f32>) -> Array2<f32> {
    let mut result = arr.clone();
    for mut row in result.rows_mut() {
        let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-12 {
            row.mapv_inplace(|x| x / norm);
        }
    }
    result
}
