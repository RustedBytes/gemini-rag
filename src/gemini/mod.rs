use std::{
    path::Path,
    time::{Duration, Instant},
};

mod types;
mod upload;

pub use types::{FileSearchStore, GenerateContentResponse, Model, Operation, RetrievedContext};

use anyhow::{Context, Result, anyhow, bail};
use base64::{Engine, engine::general_purpose::STANDARD as BASE64};
use reqwest::{
    StatusCode, Url,
    header::{HeaderMap, HeaderValue},
};
use serde::Deserialize;
use serde_json::{Map, Value, json};
use tokio::time::sleep;

use crate::logging;
use types::{ListModelsResponse, ListStoresResponse};

#[derive(Clone)]
pub struct GeminiClient {
    http: reqwest::Client,
    api_key: String,
    base_url: String,
}

impl GeminiClient {
    pub fn new(api_key: Option<String>, base_url: String) -> Result<Self> {
        let api_key = api_key.unwrap_or_default();
        if api_key.trim().is_empty() {
            bail!("GEMINI_API_KEY or --api-key is required");
        }

        logging::event(format!(
            "gemini client initialized: base_url={} api_key_present=true api_key_len={}",
            base_url.trim_end_matches('/'),
            api_key.trim().len()
        ));

        Ok(Self {
            http: reqwest::Client::new(),
            api_key: api_key.trim().to_string(),
            base_url: base_url.trim_end_matches('/').to_string(),
        })
    }

    pub async fn create_store(
        &self,
        display_name: &str,
        embedding_model: Option<&str>,
    ) -> Result<FileSearchStore> {
        logging::event(format!(
            "create file search store: display_name={display_name} embedding_model={}",
            embedding_model.unwrap_or("<default>")
        ));
        let url = self.url("/v1beta/fileSearchStores");
        let mut body = json!({ "displayName": display_name });
        if let Some(embedding_model) = embedding_model {
            body["embeddingModel"] = json!(embedding_model);
        }
        logging::debug(format!("POST {url} body={body}"));
        let response = self
            .http
            .post(url)
            .query(&[("key", &self.api_key)])
            .json(&body)
            .send()
            .await
            .context("failed to create file search store")?;
        logging::event(format!(
            "create file search store response: status={}",
            response.status()
        ));

        self.json_response(response).await
    }

    pub async fn list_stores(&self) -> Result<Vec<FileSearchStore>> {
        logging::event("list file search stores");
        let url = self.url("/v1beta/fileSearchStores");
        logging::debug(format!("GET {url}"));
        let response = self
            .http
            .get(url)
            .query(&[("key", &self.api_key)])
            .send()
            .await
            .context("failed to list file search stores")?;
        logging::event(format!(
            "list file search stores response: status={}",
            response.status()
        ));

        Ok(self
            .json_response::<ListStoresResponse>(response)
            .await?
            .file_search_stores)
    }

    pub async fn get_store(&self, store: &str) -> Result<FileSearchStore> {
        logging::event(format!("get file search store: store={store}"));
        let url = self.url(&format!("/v1beta/{store}"));
        logging::debug(format!("GET {url}"));
        let response = self
            .http
            .get(url)
            .query(&[("key", &self.api_key)])
            .send()
            .await
            .with_context(|| format!("failed to get file search store {store}"))?;
        logging::event(format!(
            "get file search store response: status={}",
            response.status()
        ));

        self.json_response(response)
            .await
            .with_context(|| format!("file search store is not accessible: {store}"))
    }

    pub async fn list_models(&self) -> Result<Vec<Model>> {
        logging::event("list models");
        let url = self.url("/v1beta/models");
        logging::debug(format!("GET {url}"));
        let response = self
            .http
            .get(url)
            .query(&[("key", &self.api_key)])
            .send()
            .await
            .context("failed to list Gemini models")?;
        logging::event(format!(
            "list models response: status={}",
            response.status()
        ));

        Ok(self
            .json_response::<ListModelsResponse>(response)
            .await?
            .models)
    }

    pub async fn delete_store(&self, store: &str, force: bool) -> Result<()> {
        logging::event(format!(
            "delete file search store: store={store} force={force}"
        ));
        let url = self.url(&format!("/v1beta/{store}"));
        logging::debug(format!("DELETE {url} force={force}"));
        let response = self
            .http
            .delete(url)
            .query(&[
                ("key", self.api_key.as_str()),
                ("force", &force.to_string()),
            ])
            .send()
            .await
            .with_context(|| format!("failed to delete {store}"))?;
        logging::event(format!(
            "delete file search store response: status={}",
            response.status()
        ));

        self.empty_response(response).await
    }

    pub async fn delete_document(&self, store: &str, document: &str) -> Result<()> {
        let document_name = document_resource_name(store, document);
        logging::event(format!(
            "delete file search document: document={document_name} force=true"
        ));
        let url = self.url(&format!("/v1beta/{document_name}"));
        logging::debug(format!("DELETE {url} force=true"));
        let response = self
            .http
            .delete(url)
            .query(&[("key", self.api_key.as_str()), ("force", "true")])
            .send()
            .await
            .with_context(|| format!("failed to delete document {document_name}"))?;
        logging::event(format!(
            "delete file search document response: document={document_name} status={}",
            response.status()
        ));

        self.empty_response(response).await
    }

    pub async fn wait_for_operation(
        &self,
        operation: Operation,
        poll_interval: Duration,
        timeout: Option<Duration>,
    ) -> Result<()> {
        logging::event(format!("wait operation: operation={}", operation.name));
        let mut operation = operation;
        let started_at = Instant::now();
        loop {
            if let Some(error) = operation.error {
                logging::event(format!(
                    "operation failed: operation={} code={:?} message={:?}",
                    operation.name, error.code, error.message
                ));
                bail!(
                    "operation {} failed: {}",
                    operation.name,
                    error
                        .message
                        .unwrap_or_else(|| format!("status code {:?}", error.code))
                );
            }

            if operation.done {
                logging::event(format!("operation done: operation={}", operation.name));
                return Ok(());
            }

            if let Some(timeout) = timeout
                && started_at.elapsed() >= timeout
            {
                logging::event(format!(
                    "operation timed out: operation={} elapsed_secs={} timeout_secs={}",
                    operation.name,
                    started_at.elapsed().as_secs(),
                    timeout.as_secs()
                ));
                bail!(
                    "operation {} did not finish within {}s; it may still complete server-side. Retry later, increase --operation-timeout-secs, or use --no-wait.",
                    operation.name,
                    timeout.as_secs()
                );
            }

            logging::debug(format!(
                "operation pending: operation={} elapsed_secs={} timeout_secs={:?} sleeping_ms={}",
                operation.name,
                started_at.elapsed().as_secs(),
                timeout.map(|timeout| timeout.as_secs()),
                poll_interval.as_millis()
            ));
            sleep(poll_interval).await;
            operation = self.get_operation(&operation.name).await?;
        }
    }

    pub async fn generate_content(
        &self,
        model: &str,
        store: &str,
        prompt: &str,
        system_prompt: Option<&str>,
    ) -> Result<GenerateContentResponse> {
        self.generate_content_with_optional_store(model, Some(store), prompt, system_prompt, &[])
            .await
    }

    pub async fn generate_content_with_optional_store(
        &self,
        model: &str,
        store: Option<&str>,
        prompt: &str,
        system_prompt: Option<&str>,
        response_modalities: &[String],
    ) -> Result<GenerateContentResponse> {
        let model = model.strip_prefix("models/").unwrap_or(model);
        logging::event(format!(
            "generate content: model={model} store={} prompt_chars={} system_prompt_chars={} response_modalities={}",
            store.unwrap_or("<none>"),
            prompt.chars().count(),
            system_prompt
                .map(str::chars)
                .map(Iterator::count)
                .unwrap_or(0),
            response_modalities.join(",")
        ));
        let url = self.url(&format!("/v1beta/models/{model}:generateContent"));
        let body = generate_content_body(prompt, system_prompt, response_modalities, store);
        logging::debug(format!("POST {url} generateContent body={body}"));
        let response = self
            .http
            .post(url)
            .query(&[("key", &self.api_key)])
            .json(&body)
            .send()
            .await
            .context("failed to query Gemini")?;
        logging::event(format!(
            "generate content response: status={}",
            response.status()
        ));

        self.json_response(response).await
    }

    pub async fn stream_generate_content_with_optional_store(
        &self,
        model: &str,
        store: Option<&str>,
        prompt: &str,
        system_prompt: Option<&str>,
        response_modalities: &[String],
    ) -> Result<reqwest::Response> {
        let model = model.strip_prefix("models/").unwrap_or(model);
        logging::event(format!(
            "stream generate content: model={model} store={} prompt_chars={} system_prompt_chars={} response_modalities={}",
            store.unwrap_or("<none>"),
            prompt.chars().count(),
            system_prompt
                .map(str::chars)
                .map(Iterator::count)
                .unwrap_or(0),
            response_modalities.join(",")
        ));
        let url = self.url(&format!("/v1beta/models/{model}:streamGenerateContent"));
        let body = generate_content_body(prompt, system_prompt, response_modalities, store);
        logging::debug(format!("POST {url} streamGenerateContent body={body}"));
        let response = self
            .http
            .post(url)
            .query(&[("key", self.api_key.as_str()), ("alt", "sse")])
            .json(&body)
            .send()
            .await
            .context("failed to stream query Gemini")?;
        let status = response.status();
        logging::event(format!("stream generate content response: status={status}"));
        if !status.is_success() {
            return Err(api_error(status, response.text().await.unwrap_or_default()));
        }

        Ok(response)
    }

    pub async fn extract_text_from_image(
        &self,
        model: &str,
        image_path: &Path,
        prompt: &str,
    ) -> Result<String> {
        let image_bytes = tokio::fs::read(image_path)
            .await
            .with_context(|| format!("failed to read {}", image_path.display()))?;
        let mime_type = mime_guess::from_path(image_path)
            .first_or_octet_stream()
            .essence_str()
            .to_string();
        let model = model.strip_prefix("models/").unwrap_or(model);
        logging::event(format!(
            "extract text from image: model={model} path={} bytes={} mime_type={mime_type}",
            image_path.display(),
            image_bytes.len()
        ));
        let url = self.url(&format!("/v1beta/models/{model}:generateContent"));
        logging::debug(format!(
            "POST {url} image OCR body: prompt_chars={} inline_bytes={}",
            prompt.chars().count(),
            image_bytes.len()
        ));
        let body = json!({
            "contents": [{
                "role": "user",
                "parts": [
                    { "text": prompt },
                    {
                        "inlineData": {
                            "mimeType": mime_type,
                            "data": BASE64.encode(image_bytes)
                        }
                    }
                ]
            }]
        });
        let response = self
            .http
            .post(url)
            .query(&[("key", &self.api_key)])
            .json(&body)
            .send()
            .await
            .with_context(|| format!("failed to OCR {}", image_path.display()))?;
        logging::event(format!(
            "extract text from image response: path={} status={}",
            image_path.display(),
            response.status()
        ));
        let response = self
            .json_response::<GenerateContentResponse>(response)
            .await
            .with_context(|| format!("failed to OCR {}", image_path.display()))?;

        response
            .text()
            .ok_or_else(|| anyhow!("Gemini OCR response did not include text"))
    }

    async fn get_operation(&self, operation_name: &str) -> Result<Operation> {
        logging::event(format!("get operation: operation={operation_name}"));
        let url = self.url(&format!("/v1beta/{operation_name}"));
        logging::debug(format!("GET {url}"));
        let response = self
            .http
            .get(url)
            .query(&[("key", &self.api_key)])
            .send()
            .await
            .with_context(|| format!("failed to poll operation {operation_name}"))?;
        logging::event(format!(
            "get operation response: status={}",
            response.status()
        ));

        self.json_response(response).await
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base_url, path)
    }

    async fn json_response<T: for<'de> Deserialize<'de>>(
        &self,
        response: reqwest::Response,
    ) -> Result<T> {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        logging::debug(format!(
            "json response received: status={status} bytes={} body={text}",
            text.len()
        ));
        if !status.is_success() {
            return Err(api_error(status, text));
        }

        serde_json::from_str(&text).with_context(|| format!("failed to parse response: {text}"))
    }

    async fn empty_response(&self, response: reqwest::Response) -> Result<()> {
        let status = response.status();
        if status.is_success() {
            logging::debug(format!("empty response success: status={status}"));
            return Ok(());
        }

        let text = response.text().await.unwrap_or_default();
        logging::debug(format!(
            "empty response error: status={status} bytes={} body={text}",
            text.len()
        ));
        Err(api_error(status, text))
    }
}

fn header_value<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
    headers
        .get(name)
        .and_then(|value: &HeaderValue| value.to_str().ok())
}

fn redacted_headers(headers: &HeaderMap) -> String {
    headers
        .iter()
        .map(|(name, value)| {
            let value = value
                .to_str()
                .map(redact_header_value)
                .unwrap_or_else(|_| "<non-utf8>".to_string());
            format!("{name}: {value}")
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn is_retryable_status(status: StatusCode) -> bool {
    status == StatusCode::REQUEST_TIMEOUT
        || status == StatusCode::TOO_MANY_REQUESTS
        || status.is_server_error()
}

fn next_backoff(current: Duration) -> Duration {
    current
        .saturating_mul(2)
        .min(upload::UPLOAD_FINALIZE_MAX_BACKOFF)
}

fn redact_header_value(value: &str) -> String {
    if let Ok(mut url) = Url::parse(value) {
        let mut redacted = false;
        let pairs = url
            .query_pairs()
            .map(|(key, value)| {
                if key == "key" {
                    redacted = true;
                    (key.into_owned(), "<redacted>".to_string())
                } else {
                    (key.into_owned(), value.into_owned())
                }
            })
            .collect::<Vec<_>>();
        if redacted {
            url.query_pairs_mut().clear().extend_pairs(pairs);
            return url.to_string();
        }
    }

    value.to_string()
}

fn generate_content_body(
    prompt: &str,
    system_prompt: Option<&str>,
    response_modalities: &[String],
    store: Option<&str>,
) -> Value {
    let mut body = Map::from_iter([(
        "contents".to_string(),
        json!([{
            "role": "user",
            "parts": [{ "text": prompt }]
        }]),
    )]);
    [
        system_prompt.map(|system_prompt| {
            (
                "systemInstruction",
                json!({
                "parts": [{ "text": system_prompt }]
                    }),
            )
        }),
        (!response_modalities.is_empty()).then(|| {
            (
                "generationConfig",
                json!({
                "responseModalities": response_modalities
                }),
            )
        }),
        store.map(|store| {
            (
                "tools",
                json!([{
                "file_search": {
                    "file_search_store_names": [store]
                }
                    }]),
            )
        }),
    ]
    .into_iter()
    .flatten()
    .for_each(|(key, value)| {
        body.insert(key.to_string(), value);
    });

    Value::Object(body)
}

fn document_resource_name(store: &str, document: &str) -> String {
    format!("{store}/documents/{document}")
}

fn api_error(status: StatusCode, body: String) -> anyhow::Error {
    logging::error(format!(
        "Gemini API error response: status={status} bytes={} body={body}",
        body.len()
    ));
    if let Ok(value) = serde_json::from_str::<Value>(&body)
        && let Some(message) = value
            .pointer("/error/message")
            .and_then(Value::as_str)
            .or_else(|| value.pointer("/message").and_then(Value::as_str))
    {
        return anyhow!("Gemini API returned {status}: {message}");
    }

    anyhow!("Gemini API returned {status}: {body}")
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use reqwest::{
        StatusCode,
        header::{HeaderMap, HeaderValue},
    };
    use serde_json::json;

    use super::{
        api_error, document_resource_name, generate_content_body, header_value,
        is_retryable_status, next_backoff, redact_header_value, redacted_headers,
    };

    #[test]
    fn document_resource_name_combines_store_and_document_id() {
        assert_eq!(
            document_resource_name("fileSearchStores/demo", "the-doc-abc"),
            "fileSearchStores/demo/documents/the-doc-abc"
        );
    }

    #[test]
    fn generate_content_body_includes_optional_sections() {
        let modalities = vec!["TEXT".to_string(), "IMAGE".to_string()];

        let body = generate_content_body(
            "Question?",
            Some("Answer carefully."),
            &modalities,
            Some("fileSearchStores/demo"),
        );

        assert_eq!(
            body,
            json!({
                "contents": [{
                    "role": "user",
                    "parts": [{ "text": "Question?" }]
                }],
                "systemInstruction": {
                    "parts": [{ "text": "Answer carefully." }]
                },
                "generationConfig": {
                    "responseModalities": ["TEXT", "IMAGE"]
                },
                "tools": [{
                    "file_search": {
                        "file_search_store_names": ["fileSearchStores/demo"]
                    }
                }]
            })
        );
    }

    #[test]
    fn generate_content_body_omits_absent_optional_sections() {
        let body = generate_content_body("Question?", None, &[], None);

        assert_eq!(
            body,
            json!({
                "contents": [{
                    "role": "user",
                    "parts": [{ "text": "Question?" }]
                }]
            })
        );
    }

    #[test]
    fn header_helpers_read_and_redact_values() {
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-goog-upload-url",
            HeaderValue::from_static("https://example.test/upload?key=secret&alt=sse"),
        );
        headers.insert(
            "x-binary",
            HeaderValue::from_bytes(b"\xff").expect("header"),
        );

        assert_eq!(
            header_value(&headers, "x-goog-upload-url"),
            Some("https://example.test/upload?key=secret&alt=sse")
        );

        let redacted = redacted_headers(&headers);
        assert!(redacted.contains("key=%3Credacted%3E"));
        assert!(redacted.contains("x-binary: <non-utf8>"));
        assert!(!redacted.contains("secret"));
    }

    #[test]
    fn redact_header_value_leaves_non_urls_unchanged() {
        assert_eq!(redact_header_value("plain-token"), "plain-token");
    }

    #[test]
    fn retry_status_and_backoff_rules_are_conservative() {
        assert!(is_retryable_status(StatusCode::REQUEST_TIMEOUT));
        assert!(is_retryable_status(StatusCode::TOO_MANY_REQUESTS));
        assert!(is_retryable_status(StatusCode::BAD_GATEWAY));
        assert!(!is_retryable_status(StatusCode::BAD_REQUEST));

        assert_eq!(next_backoff(Duration::from_secs(2)), Duration::from_secs(4));
        assert_eq!(
            next_backoff(Duration::from_secs(20)),
            Duration::from_secs(30)
        );
    }

    #[test]
    fn api_error_extracts_structured_error_message() {
        let error = api_error(
            StatusCode::BAD_REQUEST,
            json!({
                "error": {
                    "message": "invalid request"
                }
            })
            .to_string(),
        );

        assert_eq!(
            error.to_string(),
            "Gemini API returned 400 Bad Request: invalid request"
        );
    }
}
