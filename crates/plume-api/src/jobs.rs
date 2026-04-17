use std::collections::HashMap;
use std::sync::Arc;

use plume_core::types::{IndexJobResponse, IndexJobStatus};
use tokio::sync::RwLock;
use uuid::Uuid;

#[derive(Clone, Default)]
pub struct IndexJobRegistry {
    jobs: Arc<RwLock<HashMap<String, IndexJobResponse>>>,
}

impl IndexJobRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn create_job(&self, namespace: &str, index_type: &str) -> IndexJobResponse {
        let job = IndexJobResponse {
            job_id: Uuid::new_v4().to_string(),
            namespace: namespace.to_string(),
            index_type: index_type.to_string(),
            status: IndexJobStatus::Queued,
            error: None,
        };

        self.jobs
            .write()
            .await
            .insert(job.job_id.clone(), job.clone());

        job
    }

    pub async fn get(&self, job_id: &str) -> Option<IndexJobResponse> {
        self.jobs.read().await.get(job_id).cloned()
    }

    pub async fn mark_running(&self, job_id: &str) {
        self.update(job_id, IndexJobStatus::Running, None).await;
    }

    pub async fn mark_completed(&self, job_id: &str) {
        self.update(job_id, IndexJobStatus::Completed, None).await;
    }

    pub async fn mark_failed(&self, job_id: &str, error: String) {
        self.update(job_id, IndexJobStatus::Failed, Some(error))
            .await;
    }

    async fn update(&self, job_id: &str, status: IndexJobStatus, error: Option<String>) {
        if let Some(job) = self.jobs.write().await.get_mut(job_id) {
            job.status = status;
            job.error = error;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn tracks_job_lifecycle() {
        let registry = IndexJobRegistry::new();

        let job = registry.create_job("code", "vector").await;
        assert_eq!(job.status, IndexJobStatus::Queued);

        registry.mark_running(&job.job_id).await;
        assert_eq!(
            registry.get(&job.job_id).await.unwrap().status,
            IndexJobStatus::Running
        );

        registry.mark_completed(&job.job_id).await;
        let completed = registry.get(&job.job_id).await.unwrap();
        assert_eq!(completed.status, IndexJobStatus::Completed);
        assert_eq!(completed.error, None);
    }

    #[tokio::test]
    async fn stores_failure_details() {
        let registry = IndexJobRegistry::new();

        let job = registry.create_job("code", "fts").await;
        registry
            .mark_failed(&job.job_id, "index build failed".to_string())
            .await;

        let failed = registry.get(&job.job_id).await.unwrap();
        assert_eq!(failed.status, IndexJobStatus::Failed);
        assert_eq!(failed.error.as_deref(), Some("index build failed"));
    }
}
