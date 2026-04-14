use std::collections::HashMap;
use std::sync::RwLock;

/// Per-namespace generation counter for O(1) cache invalidation.
///
/// Every write (upsert) to a namespace increments its generation.
/// Cache keys include the generation, so stale entries are never matched —
/// they simply age out of the LFU cache naturally.
pub struct GenerationCounter {
    counters: RwLock<HashMap<String, u64>>,
}

impl GenerationCounter {
    pub fn new() -> Self {
        Self {
            counters: RwLock::new(HashMap::new()),
        }
    }

    /// Get the current generation for a namespace (0 if never written).
    pub fn get(&self, namespace: &str) -> u64 {
        self.counters
            .read()
            .unwrap()
            .get(namespace)
            .copied()
            .unwrap_or(0)
    }

    /// Increment the generation counter, returning the new value.
    pub fn increment(&self, namespace: &str) -> u64 {
        let mut counters = self.counters.write().unwrap();
        let counter = counters.entry(namespace.to_string()).or_insert(0);
        *counter += 1;
        *counter
    }

    /// Remove the counter for a dropped namespace.
    pub fn remove(&self, namespace: &str) {
        self.counters.write().unwrap().remove(namespace);
    }
}

impl Default for GenerationCounter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generation_counter() {
        let gc = GenerationCounter::new();
        assert_eq!(gc.get("ns1"), 0);

        assert_eq!(gc.increment("ns1"), 1);
        assert_eq!(gc.increment("ns1"), 2);
        assert_eq!(gc.get("ns1"), 2);

        // Other namespace unaffected
        assert_eq!(gc.get("ns2"), 0);

        gc.remove("ns1");
        assert_eq!(gc.get("ns1"), 0);
    }
}
