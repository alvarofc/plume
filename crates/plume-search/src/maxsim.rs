use plume_core::types::MultiVector;

/// Compute the MaxSim score between a query and a document multi-vector.
///
/// For each query token vector, find the maximum cosine similarity with any
/// document token vector. The final score is the sum of these maximums.
///
/// This is the core ColBERT late-interaction scoring mechanism: it allows
/// fine-grained token-level matching without needing to see query and document
/// together at encoding time.
pub fn maxsim_score(query: &MultiVector, document: &MultiVector) -> f32 {
    if query.is_empty() || document.is_empty() {
        return 0.0;
    }

    let mut total = 0.0f32;

    for q_vec in query {
        let mut max_sim = f32::NEG_INFINITY;
        for d_vec in document {
            let sim = cosine_similarity(q_vec, d_vec);
            if sim > max_sim {
                max_sim = sim;
            }
        }
        total += max_sim;
    }

    total
}

/// Cosine similarity between two vectors (assumed L2-normalized → dot product).
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_maxsim_identical() {
        let v = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
        let score = maxsim_score(&v, &v);
        assert!((score - 2.0).abs() < 1e-6, "identical vectors should score 2.0, got {score}");
    }

    #[test]
    fn test_maxsim_orthogonal() {
        let query = vec![vec![1.0, 0.0]];
        let doc = vec![vec![0.0, 1.0]];
        let score = maxsim_score(&query, &doc);
        assert!((score - 0.0).abs() < 1e-6, "orthogonal should score 0.0, got {score}");
    }

    #[test]
    fn test_maxsim_picks_best_match() {
        let query = vec![vec![1.0, 0.0, 0.0]];
        let doc = vec![
            vec![0.0, 1.0, 0.0],  // orthogonal
            vec![0.8, 0.6, 0.0],  // partial match
            vec![1.0, 0.0, 0.0],  // perfect match
        ];
        let score = maxsim_score(&query, &doc);
        assert!((score - 1.0).abs() < 1e-6, "should pick perfect match, got {score}");
    }

    #[test]
    fn test_maxsim_empty() {
        assert_eq!(maxsim_score(&vec![], &vec![vec![1.0]]), 0.0);
        assert_eq!(maxsim_score(&vec![vec![1.0]], &vec![]), 0.0);
    }
}
