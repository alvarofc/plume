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
    /// NVMe/disk cache capacity for the persistent L2 tier.
    #[serde(default = "default_nvme_capacity_gb")]
    pub nvme_capacity_gb: usize,
    /// Filesystem path used by the persistent L2 tier.
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
    #[serde(default)]
    pub num_partitions: Option<u32>,
    #[serde(default = "default_nprobes")]
    pub nprobes: u32,
    #[serde(default)]
    pub refine_factor: Option<u32>,
    #[serde(default = "default_ann_candidate_multiplier")]
    pub ann_candidate_multiplier: usize,
    /// Hard ceiling on ANN candidates materialized per query. Every
    /// candidate pulls its full multivector into memory for MaxSim
    /// rerank, so without a cap a pathological `k` combined with a
    /// large `ann_candidate_multiplier` could OOM the process.
    #[serde(default = "default_max_candidates")]
    pub max_candidates: usize,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            nbits: default_nbits(),
            num_partitions: None,
            nprobes: default_nprobes(),
            refine_factor: None,
            ann_candidate_multiplier: default_ann_candidate_multiplier(),
            max_candidates: default_max_candidates(),
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

    pub fn from_env_or_default() -> Result<Self, crate::error::PlumeError> {
        match std::env::var("PLUME_CONFIG") {
            Ok(path) => Self::from_file(&path),
            Err(_) => Ok(Self::default()),
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
    "./data/lancedb".into()
}
fn default_ram_capacity_mb() -> usize {
    2048
}
fn default_nvme_capacity_gb() -> usize {
    50
}
fn default_nvme_path() -> String {
    "./data/plume-cache".into()
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
    8
}
fn default_nprobes() -> u32 {
    20
}
fn default_ann_candidate_multiplier() -> usize {
    50
}
fn default_max_candidates() -> usize {
    // Bounded at ~2x the default multiplier*k=50*100=5000; operators
    // can raise this in `config.toml` for large-corpus recall sweeps.
    10_000
}
