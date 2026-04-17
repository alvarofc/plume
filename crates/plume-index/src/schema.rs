use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{
    ArrayRef, FixedSizeListArray, Float32Array, LargeStringArray, ListArray, RecordBatch,
    StringArray,
};
use arrow_buffer::{OffsetBuffer, ScalarBuffer};
use arrow_schema::{DataType, Field, Schema};
use plume_core::error::PlumeError;
use plume_core::types::{MultiVector, EMBEDDING_DIM};

/// Build the Arrow schema for a Plume namespace table.
///
/// Schema:
///   id: Utf8 (primary key)
///   text: LargeUtf8 (full document text, for BM25)
///   multivector: List<FixedSizeList<Float32, EMBEDDING_DIM>> (ColBERT token embeddings)
///   metadata: LargeUtf8 (JSON string)
///
/// LanceDB indexes the `multivector` column natively with IVF_PQ + MaxSim
/// late interaction — no pooled single-vector column is needed.
pub fn plume_schema() -> Schema {
    let embedding_field = Field::new("item", DataType::Float32, false);
    let token_vec = DataType::FixedSizeList(Arc::new(embedding_field), EMBEDDING_DIM as i32);
    let multivector_type = DataType::List(Arc::new(Field::new("item", token_vec, true)));

    Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("text", DataType::LargeUtf8, false),
        Field::new("multivector", multivector_type, false),
        Field::new("metadata", DataType::LargeUtf8, true),
    ])
}

/// Build a RecordBatch from documents and their multi-vector embeddings.
pub fn build_record_batch(
    ids: &[String],
    texts: &[String],
    multivectors: &[MultiVector],
    metadata: &[HashMap<String, serde_json::Value>],
) -> Result<RecordBatch, PlumeError> {
    let schema = Arc::new(plume_schema());

    let id_array = StringArray::from(ids.to_vec());
    let text_array = LargeStringArray::from(texts.to_vec());
    let mv_array = build_multivector_array(multivectors)?;

    let meta_strings: Vec<String> = metadata
        .iter()
        .map(|m| {
            serde_json::to_string(m)
                .map_err(|e| PlumeError::Index(format!("metadata serialization failed: {e}")))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let meta_array = LargeStringArray::from(meta_strings);

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(id_array),
            Arc::new(text_array),
            mv_array,
            Arc::new(meta_array),
        ],
    )
    .map_err(|e| PlumeError::Index(format!("failed to build record batch: {e}")))
}

/// Build the nested List<FixedSizeList<Float32, EMBEDDING_DIM>> array for multivectors.
fn build_multivector_array(multivectors: &[MultiVector]) -> Result<ArrayRef, PlumeError> {
    let mut all_values: Vec<f32> = Vec::new();
    let mut list_offsets: Vec<i32> = vec![0];
    let mut total_tokens: i32 = 0;

    for mv in multivectors {
        for token_vec in mv {
            if token_vec.len() != EMBEDDING_DIM {
                return Err(PlumeError::Index(format!(
                    "expected embedding dim {EMBEDDING_DIM}, got {}",
                    token_vec.len()
                )));
            }
            all_values.extend_from_slice(token_vec);
            total_tokens += 1;
        }
        list_offsets.push(total_tokens);
    }

    let values = Float32Array::from(all_values);

    let fsl_field = Arc::new(Field::new("item", DataType::Float32, false));
    let fsl_array =
        FixedSizeListArray::new(fsl_field, EMBEDDING_DIM as i32, Arc::new(values), None);

    let inner_field = Arc::new(Field::new(
        "item",
        DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, false)),
            EMBEDDING_DIM as i32,
        ),
        true,
    ));

    let offsets = OffsetBuffer::new(ScalarBuffer::from(list_offsets));
    let list_array = ListArray::new(inner_field, offsets, Arc::new(fsl_array), None);

    Ok(Arc::new(list_array))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_batch_has_multivector_column() {
        let ids = vec!["1".to_string()];
        let texts = vec!["hello".to_string()];
        let multivectors = vec![vec![vec![1.0; EMBEDDING_DIM]]];
        let metadata = vec![HashMap::new()];

        let batch = build_record_batch(&ids, &texts, &multivectors, &metadata).expect("batch");

        assert!(batch.column_by_name("multivector").is_some());
        assert!(batch.column_by_name("ann_vector").is_none());
        assert_eq!(batch.num_columns(), 4);
    }
}
