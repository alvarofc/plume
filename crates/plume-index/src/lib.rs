mod schema;
mod table;

use std::collections::HashMap;
use std::sync::Arc;

use lancedb::Connection;
use plume_core::config::StorageConfig;
use plume_core::error::PlumeError;
use tokio::sync::RwLock;
use tracing::info;

pub use schema::build_record_batch;
pub use table::NamespaceTable;

/// Index manager: manages LanceDB connections and per-namespace tables.
pub struct IndexManager {
    connection: Connection,
    tables: RwLock<HashMap<String, Arc<NamespaceTable>>>,
}

impl IndexManager {
    /// Connect to LanceDB at the given storage URI.
    pub async fn connect(config: &StorageConfig) -> Result<Self, PlumeError> {
        let builder = lancedb::connect(&config.uri);

        let connection = builder
            .execute()
            .await
            .map_err(|e| PlumeError::Index(format!("failed to connect to LanceDB: {e}")))?;

        info!(uri = %config.uri, "connected to LanceDB");

        Ok(Self {
            connection,
            tables: RwLock::new(HashMap::new()),
        })
    }

    /// Get or create a namespace table.
    pub async fn namespace(&self, name: &str) -> Result<Arc<NamespaceTable>, PlumeError> {
        {
            let tables = self.tables.read().await;
            if let Some(table) = tables.get(name) {
                return Ok(Arc::clone(table));
            }
        }

        let ns_table = NamespaceTable::open_or_create(&self.connection, name).await?;
        let ns_table = Arc::new(ns_table);

        let mut tables = self.tables.write().await;
        tables.insert(name.to_string(), Arc::clone(&ns_table));

        Ok(ns_table)
    }

    /// Drop a namespace.
    pub async fn drop_namespace(&self, name: &str) -> Result<(), PlumeError> {
        self.connection
            .drop_table(name, &[])
            .await
            .map_err(|e| PlumeError::Index(format!("failed to drop table {name}: {e}")))?;

        let mut tables = self.tables.write().await;
        tables.remove(name);

        info!(namespace = name, "namespace dropped");
        Ok(())
    }

    /// List all namespace names.
    pub async fn list_namespaces(&self) -> Result<Vec<String>, PlumeError> {
        self.connection
            .table_names()
            .execute()
            .await
            .map_err(|e| PlumeError::Index(format!("failed to list tables: {e}")))
    }
}
