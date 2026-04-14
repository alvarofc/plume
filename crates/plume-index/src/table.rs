use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{Array, RecordBatch, RecordBatchIterator};
use futures::StreamExt;
use lancedb::index::scalar::FullTextSearchQuery;
use lancedb::query::{ExecutableQuery, QueryBase, Select};
use lancedb::{Connection, Table};
use plume_core::error::PlumeError;
use plume_core::types::{MultiVector, SearchResult};
use tokio::sync::RwLock;
use tracing::info;

use crate::schema::{build_record_batch, plume_schema};

/// A namespace-scoped LanceDB table.
pub struct NamespaceTable {
    pub name: String,
    table: RwLock<Table>,
    /// Track whether the table has been written to, to avoid count_rows on every upsert.
    has_rows: std::sync::atomic::AtomicBool,
}

impl NamespaceTable {
    /// Open an existing table (returns error if it doesn't exist).
    pub async fn open(conn: &Connection, name: &str) -> Result<Self, PlumeError> {
        let table = conn
            .open_table(name)
            .execute()
            .await
            .map_err(|e| PlumeError::Index(format!("table {name} not found: {e}")))?;

        Ok(Self {
            name: name.to_string(),
            table: RwLock::new(table),
            has_rows: std::sync::atomic::AtomicBool::new(true),
        })
    }

    /// Open an existing table or create a new one.
    pub async fn open_or_create(conn: &Connection, name: &str) -> Result<Self, PlumeError> {
        let (table, has_data) = match conn.open_table(name).execute().await {
            Ok(t) => {
                info!(namespace = name, "opened existing table");
                (t, true)
            }
            Err(_) => {
                let schema = Arc::new(plume_schema());
                let t = conn
                    .create_empty_table(name, schema)
                    .execute()
                    .await
                    .map_err(|e| {
                        PlumeError::Index(format!("failed to create table {name}: {e}"))
                    })?;
                (t, false)
            }
        };

        Ok(Self {
            name: name.to_string(),
            table: RwLock::new(table),
            has_rows: std::sync::atomic::AtomicBool::new(has_data),
        })
    }

    /// Upsert documents with their multivector embeddings.
    pub async fn upsert(
        &self,
        ids: &[String],
        texts: &[String],
        multivectors: &[MultiVector],
        metadata: &[HashMap<String, serde_json::Value>],
    ) -> Result<usize, PlumeError> {
        let batch = build_record_batch(ids, texts, multivectors, metadata)?;
        let count = batch.num_rows();

        let table = self.table.write().await;

        if !self.has_rows.load(std::sync::atomic::Ordering::Relaxed) {
            // Empty table: use simple add (merge_insert panics on empty tables)
            table
                .add(vec![batch])
                .execute()
                .await
                .map_err(|e| PlumeError::Index(format!("insert failed: {e}")))?;
            self.has_rows
                .store(true, std::sync::atomic::Ordering::Relaxed);
        } else {
            // Table has data: use merge-insert for upsert semantics
            let schema = Arc::new(crate::schema::plume_schema());
            let reader = Box::new(RecordBatchIterator::new(vec![Ok(batch)], schema));
            let mut merge = table.merge_insert(&["id"]);
            merge.when_matched_update_all(None);
            merge.when_not_matched_insert_all();
            merge
                .execute(reader)
                .await
                .map_err(|e| PlumeError::Index(format!("upsert failed: {e}")))?;
        }

        info!(namespace = %self.name, count, "upserted documents");
        Ok(count)
    }

    /// Retrieve all documents with their multivectors for MaxSim scoring.
    pub async fn scan_with_vectors(
        &self,
        limit: usize,
    ) -> Result<Vec<(SearchResult, MultiVector)>, PlumeError> {
        use arrow_array::FixedSizeListArray;

        let table = self.table.read().await;

        let mut stream = table
            .query()
            .select(Select::columns(&["id", "text", "metadata", "multivector"]))
            .limit(limit)
            .execute()
            .await
            .map_err(|e| PlumeError::Index(format!("scan failed: {e}")))?;

        let mut results = Vec::new();

        while let Some(batch_result) = stream.next().await {
            let batch = batch_result
                .map_err(|e| PlumeError::Index(format!("scan stream error: {e}")))?;

            let ids = batch
                .column_by_name("id")
                .and_then(|c| c.as_any().downcast_ref::<arrow_array::StringArray>())
                .ok_or_else(|| PlumeError::Index("missing id column".into()))?;
            let texts = batch
                .column_by_name("text")
                .and_then(|c| c.as_any().downcast_ref::<arrow_array::LargeStringArray>())
                .ok_or_else(|| PlumeError::Index("missing text column".into()))?;
            let meta_col = batch.column_by_name("metadata");
            let mv_col = batch
                .column_by_name("multivector")
                .and_then(|c| c.as_any().downcast_ref::<arrow_array::ListArray>())
                .ok_or_else(|| PlumeError::Index("missing multivector column".into()))?;

            for i in 0..batch.num_rows() {
                let id = ids.value(i).to_string();
                let text = texts.value(i).to_string();
                let metadata: HashMap<String, serde_json::Value> = meta_col
                    .and_then(|c| {
                        c.as_any()
                            .downcast_ref::<arrow_array::LargeStringArray>()
                            .map(|a| a.value(i))
                    })
                    .and_then(|s| serde_json::from_str(s).ok())
                    .unwrap_or_default();

                // Extract multivector: List<FixedSizeList<Float32, 128>>
                let inner_array = mv_col.value(i);
                let fsl = inner_array
                    .as_any()
                    .downcast_ref::<FixedSizeListArray>()
                    .ok_or_else(|| PlumeError::Index("multivector inner type mismatch".into()))?;

                let mut mv = Vec::with_capacity(fsl.len());
                for j in 0..fsl.len() {
                    let vec_array = fsl.value(j);
                    let floats = vec_array
                        .as_any()
                        .downcast_ref::<arrow_array::Float32Array>()
                        .ok_or_else(|| PlumeError::Index("vector element type mismatch".into()))?;
                    let vec: Vec<f32> = (0..floats.len()).map(|k| floats.value(k)).collect();
                    mv.push(vec);
                }

                let result = SearchResult {
                    id,
                    text,
                    score: 0.0,
                    metadata,
                };
                results.push((result, mv));
            }
        }

        Ok(results)
    }

    /// Run a full-text (BM25) search.
    pub async fn fts_search(
        &self,
        query: &str,
        k: usize,
    ) -> Result<Vec<SearchResult>, PlumeError> {
        let table = self.table.read().await;

        let fts_query = FullTextSearchQuery::new(query.to_string());

        let results = table
            .query()
            .full_text_search(fts_query)
            .select(Select::columns(&["id", "text", "metadata"]))
            .limit(k)
            .execute()
            .await
            .map_err(|e| PlumeError::Index(format!("FTS search failed: {e}")))?;

        parse_search_results(results).await
    }

    /// Build an IVF_PQ vector index for faster search.
    pub async fn build_vector_index(&self) -> Result<(), PlumeError> {
        let table = self.table.write().await;

        table
            .create_index(&["multivector"], lancedb::index::Index::Auto)
            .execute()
            .await
            .map_err(|e| PlumeError::Index(format!("vector index build failed: {e}")))?;

        info!(namespace = %self.name, "vector index built");
        Ok(())
    }

    /// Build a BM25 full-text search index on the text column.
    pub async fn build_fts_index(&self) -> Result<(), PlumeError> {
        let table = self.table.write().await;

        table
            .create_index(
                &["text"],
                lancedb::index::Index::FTS(Default::default()),
            )
            .execute()
            .await
            .map_err(|e| PlumeError::Index(format!("FTS index build failed: {e}")))?;

        info!(namespace = %self.name, "FTS index built");
        Ok(())
    }

    /// Count total documents.
    pub async fn count(&self) -> Result<usize, PlumeError> {
        let table = self.table.read().await;
        table
            .count_rows(None)
            .await
            .map_err(|e| PlumeError::Index(format!("count failed: {e}")))
    }
}

/// Parse Arrow RecordBatch stream into SearchResults.
async fn parse_search_results(
    mut stream: impl futures::Stream<Item = Result<RecordBatch, lancedb::Error>> + Unpin,
) -> Result<Vec<SearchResult>, PlumeError> {
    let mut results = Vec::new();

    while let Some(batch_result) = stream.next().await {
        let batch = batch_result
            .map_err(|e| PlumeError::Index(format!("result stream error: {e}")))?;

        let ids = batch
            .column_by_name("id")
            .and_then(|c| c.as_any().downcast_ref::<arrow_array::StringArray>())
            .ok_or_else(|| PlumeError::Index("missing id column".into()))?;
        let texts = batch
            .column_by_name("text")
            .and_then(|c| c.as_any().downcast_ref::<arrow_array::LargeStringArray>())
            .ok_or_else(|| PlumeError::Index("missing text column".into()))?;
        let meta_col = batch.column_by_name("metadata");

        let fts_score_col = batch
            .column_by_name("_score")
            .and_then(|c| {
                c.as_any()
                    .downcast_ref::<arrow_array::Float32Array>()
                    .cloned()
            });

        for i in 0..batch.num_rows() {
            let id = ids.value(i).to_string();
            let text = texts.value(i).to_string();

            let score = if let Some(ref sc) = fts_score_col {
                sc.value(i)
            } else {
                0.0
            };

            let metadata: HashMap<String, serde_json::Value> = meta_col
                .and_then(|c| {
                    c.as_any()
                        .downcast_ref::<arrow_array::LargeStringArray>()
                        .map(|a| a.value(i))
                })
                .and_then(|s| serde_json::from_str(s).ok())
                .unwrap_or_default();

            results.push(SearchResult {
                id,
                text,
                score,
                metadata,
            });
        }
    }

    Ok(results)
}
