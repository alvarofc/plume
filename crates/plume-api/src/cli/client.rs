//! Thin HTTP client shared across CLI subcommands.

use std::time::Duration;

use anyhow::{Context, Result};
use reqwest::Response;
use serde::de::DeserializeOwned;
use serde::Serialize;

pub const DEFAULT_URL: &str = "http://localhost:8787";

/// Fail fast when the server is down. 5s is generous for localhost and
/// still quick enough that a typo'd host doesn't hang the CLI.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
/// Cap for any single request. Ingest batches that encode a fresh corpus
/// with a real ONNX model on CPU can take many minutes (the daemon will
/// happily keep encoding even after the client hangs up), so the cap has
/// to accommodate the slowest realistic upsert, not the median.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(1800);

pub struct Client {
    base_url: String,
    http: reqwest::Client,
}

impl Client {
    pub fn new(base_url: String) -> Self {
        let base_url = base_url.trim_end_matches('/').to_string();
        Self {
            base_url,
            http: reqwest::Client::builder()
                .user_agent(concat!("plume-cli/", env!("CARGO_PKG_VERSION")))
                .connect_timeout(CONNECT_TIMEOUT)
                .timeout(REQUEST_TIMEOUT)
                .build()
                .expect("reqwest client"),
        }
    }

    pub async fn get_json<T: DeserializeOwned>(&self, path: &str) -> Result<T> {
        let url = format!("{}{path}", self.base_url);
        let resp = self
            .http
            .get(&url)
            .send()
            .await
            .with_context(|| format!("GET {url}"))?;
        parse(resp).await
    }

    pub async fn post_json<B: Serialize, T: DeserializeOwned>(
        &self,
        path: &str,
        body: &B,
    ) -> Result<T> {
        let url = format!("{}{path}", self.base_url);
        let resp = self
            .http
            .post(&url)
            .json(body)
            .send()
            .await
            .with_context(|| format!("POST {url}"))?;
        parse(resp).await
    }

    pub async fn post_empty<T: DeserializeOwned>(&self, path: &str) -> Result<T> {
        let url = format!("{}{path}", self.base_url);
        let resp = self
            .http
            .post(&url)
            .send()
            .await
            .with_context(|| format!("POST {url}"))?;
        parse(resp).await
    }

    pub async fn delete_json<T: DeserializeOwned>(&self, path: &str) -> Result<T> {
        let url = format!("{}{path}", self.base_url);
        let resp = self
            .http
            .delete(&url)
            .send()
            .await
            .with_context(|| format!("DELETE {url}"))?;
        parse(resp).await
    }

    /// DELETE with a JSON body. DELETE+payload is non-standard enough
    /// that `delete_json` above doesn't bother, but bulk prune-by-id
    /// needs it.
    pub async fn delete_with_body<B: Serialize, T: DeserializeOwned>(
        &self,
        path: &str,
        body: &B,
    ) -> Result<T> {
        let url = format!("{}{path}", self.base_url);
        let resp = self
            .http
            .delete(&url)
            .json(body)
            .send()
            .await
            .with_context(|| format!("DELETE {url}"))?;
        parse(resp).await
    }
}

async fn parse<T: DeserializeOwned>(resp: Response) -> Result<T> {
    let status = resp.status();
    let body = resp.text().await.context("read response body")?;
    if !status.is_success() {
        anyhow::bail!("request failed ({status}): {body}");
    }
    serde_json::from_str(&body).with_context(|| format!("decode response: {body}"))
}
