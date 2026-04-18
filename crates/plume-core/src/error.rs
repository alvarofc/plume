use thiserror::Error;

#[derive(Debug, Error)]
pub enum PlumeError {
    #[error("encoder error: {0}")]
    Encoder(String),

    #[error("index error: {0}")]
    Index(String),

    #[error("cache error: {0}")]
    Cache(String),

    #[error("config error: {0}")]
    Config(String),

    #[error("namespace not found: {0}")]
    NamespaceNotFound(String),

    #[error("not found: {0}")]
    NotFound(String),

    #[error("invalid request: {0}")]
    InvalidRequest(String),

    #[error("internal error: {0}")]
    Internal(String),

    #[error(transparent)]
    Anyhow(#[from] anyhow::Error),
}

impl PlumeError {
    pub fn status_code(&self) -> u16 {
        match self {
            PlumeError::NamespaceNotFound(_) | PlumeError::NotFound(_) => 404,
            PlumeError::InvalidRequest(_) => 400,
            _ => 500,
        }
    }
}
