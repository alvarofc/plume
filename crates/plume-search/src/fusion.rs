use std::collections::HashMap;

use plume_core::types::SearchResult;

/// Reciprocal Rank Fusion (RRF) — merges two ranked result lists into one.
///
/// For each result appearing in either list, its RRF score is:
///   score = sum over lists of 1 / (k + rank)
///
/// where k=60 is the standard constant. This gives balanced fusion without
/// needing score normalization across different retrieval methods.
pub fn rrf_fusion(semantic: &[SearchResult], fts: &[SearchResult], k: usize) -> Vec<SearchResult> {
    const RRF_K: f32 = 60.0;

    // Map: document id → (rrf_score, best SearchResult)
    let mut scores: HashMap<String, (f32, SearchResult)> = HashMap::new();

    // Score from semantic results
    for (rank, result) in semantic.iter().enumerate() {
        let rrf_score = 1.0 / (RRF_K + rank as f32 + 1.0);
        scores
            .entry(result.id.clone())
            .and_modify(|(s, _)| *s += rrf_score)
            .or_insert((rrf_score, result.clone()));
    }

    // Score from FTS results
    for (rank, result) in fts.iter().enumerate() {
        let rrf_score = 1.0 / (RRF_K + rank as f32 + 1.0);
        scores
            .entry(result.id.clone())
            .and_modify(|(s, _)| *s += rrf_score)
            .or_insert((rrf_score, result.clone()));
    }

    // Sort by fused score descending, take top k
    let mut fused: Vec<(f32, SearchResult)> = scores.into_values().collect();
    fused.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    fused
        .into_iter()
        .take(k)
        .map(|(score, mut result)| {
            result.score = score;
            result
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_result(id: &str, score: f32) -> SearchResult {
        SearchResult {
            id: id.to_string(),
            text: format!("text for {id}"),
            score,
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_rrf_both_lists() {
        let semantic = vec![
            make_result("a", 0.9),
            make_result("b", 0.8),
            make_result("c", 0.7),
        ];
        let fts = vec![
            make_result("b", 5.0),
            make_result("d", 4.0),
            make_result("a", 3.0),
        ];

        let fused = rrf_fusion(&semantic, &fts, 10);

        // "a" appears rank 1 in semantic, rank 3 in fts → highest combined
        // "b" appears rank 2 in semantic, rank 1 in fts → also high
        assert!(!fused.is_empty());
        // Both "a" and "b" should appear in top results with scores from both lists
        let ids: Vec<&str> = fused.iter().map(|r| r.id.as_str()).collect();
        assert!(ids.contains(&"a"));
        assert!(ids.contains(&"b"));
    }

    #[test]
    fn test_rrf_k_limit() {
        let semantic = vec![make_result("a", 1.0), make_result("b", 0.9)];
        let fts = vec![make_result("c", 5.0), make_result("d", 4.0)];

        let fused = rrf_fusion(&semantic, &fts, 2);
        assert_eq!(fused.len(), 2);
    }

    #[test]
    fn test_rrf_empty() {
        let fused = rrf_fusion(&[], &[], 10);
        assert!(fused.is_empty());
    }
}
