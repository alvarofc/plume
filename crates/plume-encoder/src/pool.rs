use ndarray::Array2;

/// Pool adjacent token vectors by averaging groups of `factor` consecutive rows.
///
/// This halves (or more) the storage per document at a small quality tradeoff.
/// From NextPlaid: pool_factor=2 averages pairs of adjacent token embeddings.
pub fn pool_vectors(embeddings: &Array2<f32>, factor: usize) -> Array2<f32> {
    if factor <= 1 {
        return embeddings.clone();
    }

    let (n_tokens, dim) = embeddings.dim();
    let n_pooled = (n_tokens + factor - 1) / factor;
    let mut pooled = Array2::zeros((n_pooled, dim));

    for i in 0..n_pooled {
        let start = i * factor;
        let end = (start + factor).min(n_tokens);
        let count = (end - start) as f32;

        for j in start..end {
            for d in 0..dim {
                pooled[[i, d]] += embeddings[[j, d]] / count;
            }
        }
    }

    pooled
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_pool_factor_2() {
        let embeddings = array![
            [1.0, 2.0, 3.0],
            [3.0, 4.0, 5.0],
            [5.0, 6.0, 7.0],
            [7.0, 8.0, 9.0],
        ];

        let pooled = pool_vectors(&embeddings, 2);
        assert_eq!(pooled.dim(), (2, 3));
        assert_eq!(pooled[[0, 0]], 2.0); // (1+3)/2
        assert_eq!(pooled[[0, 1]], 3.0); // (2+4)/2
        assert_eq!(pooled[[1, 0]], 6.0); // (5+7)/2
    }

    #[test]
    fn test_pool_factor_1_noop() {
        let embeddings = array![[1.0, 2.0], [3.0, 4.0]];
        let pooled = pool_vectors(&embeddings, 1);
        assert_eq!(pooled, embeddings);
    }

    #[test]
    fn test_pool_odd_tokens() {
        let embeddings = array![
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ];

        let pooled = pool_vectors(&embeddings, 2);
        assert_eq!(pooled.dim(), (2, 2));
        assert_eq!(pooled[[0, 0]], 2.0); // (1+3)/2
        assert_eq!(pooled[[1, 0]], 5.0); // last token alone
    }
}
