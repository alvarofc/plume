use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlumeConfig {
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub storage: StorageConfig,
    #[serde(default)]
    pub cache: CacheConfig,
    #[serde(default)]
    pub encoder: EncoderConfig,
    #[serde(default)]
    pub index: IndexConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_port")]
    pub port: u16,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    #[serde(default = "default_storage_uri")]
    pub uri: String,
    pub region: Option<String>,
    pub endpoint: Option<String>,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            uri: default_storage_uri(),
            region: None,
            endpoint: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    #[serde(default = "default_ram_capacity_mb")]
    pub ram_capacity_mb: usize,
    #[serde(default = "default_nvme_capacity_gb")]
    pub nvme_capacity_gb: usize,
    #[serde(default = "default_nvme_path")]
    pub nvme_path: String,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            ram_capacity_mb: default_ram_capacity_mb(),
            nvme_capacity_gb: default_nvme_capacity_gb(),
            nvme_path: default_nvme_path(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncoderConfig {
    #[serde(default = "default_model")]
    pub model: String,
    #[serde(default = "default_pool_factor")]
    pub pool_factor: usize,
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            model: default_model(),
            pool_factor: default_pool_factor(),
            batch_size: default_batch_size(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexConfig {
    #[serde(default = "default_nbits")]
    pub nbits: u32,
    #[serde(default = "default_nprobes")]
    pub nprobes: u32,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            nbits: default_nbits(),
            nprobes: default_nprobes(),
        }
    }
}

impl PlumeConfig {
    pub fn from_file(path: &str) -> Result<Self, crate::error::PlumeError> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| crate::error::PlumeError::Config(format!("failed to read {path}: {e}")))?;
        toml::from_str(&content)
            .map_err(|e| crate::error::PlumeError::Config(format!("failed to parse config: {e}")))
    }

    pub fn from_env_or_default() -> Self {
        match std::env::var("PLUME_CONFIG") {
            Ok(path) => Self::from_file(&path).expect("failed to load config"),
            Err(_) => Self::default(),
        }
    }
}

impl Default for PlumeConfig {
    fn default() -> Self {
        Self {
            server: ServerConfig::default(),
            storage: StorageConfig::default(),
            cache: CacheConfig::default(),
            encoder: EncoderConfig::default(),
            index: IndexConfig::default(),
        }
    }
}

fn default_host() -> String {
    "0.0.0.0".into()
}
fn default_port() -> u16 {
    3000
}
fn default_storage_uri() -> String {
    "s3://plume/data".into()
}
fn default_ram_capacity_mb() -> usize {
    2048
}
fn default_nvme_capacity_gb() -> usize {
    50
}
fn default_nvme_path() -> String {
    "/var/cache/plume".into()
}
fn default_model() -> String {
    "lightonai/LateOn-Code-edge".into()
}
fn default_pool_factor() -> usize {
    2
}
fn default_batch_size() -> usize {
    32
}
fn default_nbits() -> u32 {
    4
}
fn default_nprobes() -> u32 {
    32
}
