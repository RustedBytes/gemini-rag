use std::{path::Path, time::Duration};

use anyhow::{Context, Result, anyhow, bail};
use base64::{Engine, engine::general_purpose::STANDARD as BASE64};
use reqwest::{
    StatusCode, Url,
    header::{CONTENT_LENGTH, HeaderMap, HeaderValue},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::time::sleep;

use crate::logging;

const UPLOAD_FINALIZE_MAX_ATTEMPTS: usize = 5;
const UPLOAD_FINALIZE_INITIAL_BACKOFF: Duration = Duration::from_secs(2);
const UPLOAD_FINALIZE_MAX_BACKOFF: Duration = Duration::from_secs(30);

#[derive(Clone)]
pub struct GeminiClient {
    http: reqwest::Client,
    api_key: String,
    base_url: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FileSearchStore {
    pub name: String,
    #[serde(default)]
    pub display_name: Option<String>,
    #[serde(default)]
    pub embedding_model: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ListStoresResponse {
    #[serde(default)]
    file_search_stores: Vec<FileSearchStore>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ListModelsResponse {
    #[serde(default)]
    models: Vec<Model>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Model {
    pub name: String,
    #[serde(default)]
    pub display_name: Option<String>,
    #[serde(default)]
    supported_generation_methods: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Operation {
    pub name: String,
    #[serde(default)]
    done: bool,
    #[serde(default)]
    error: Option<ApiStatus>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ApiStatus {
    #[serde(default)]
    code: Option<i32>,
    #[serde(default)]
    message: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerateContentResponse {
    #[serde(default)]
    pub candidates: Vec<Candidate>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Candidate {
    #[serde(default)]
    pub content: Option<Content>,
    #[serde(default, alias = "grounding_metadata")]
    pub grounding_metadata: Option<GroundingMetadata>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Content {
    #[serde(default)]
    pub parts: Vec<Part>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Part {
    #[serde(default)]
    pub text: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GroundingMetadata {
    #[serde(default, alias = "grounding_chunks")]
    pub grounding_chunks: Vec<GroundingChunk>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GroundingChunk {
    #[serde(default, alias = "retrieved_context")]
    pub retrieved_context: Option<RetrievedContext>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RetrievedContext {
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub uri: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct UploadMetadata<'a> {
    display_name: &'a str,
    mime_type: &'a str,
    custom_metadata: Vec<CustomMetadata<'a>>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct CustomMetadata<'a> {
    key: &'a str,
    string_value: &'a str,
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

    pub async fn upload_to_file_search_store(&self, store: &str, path: &Path) -> Result<Operation> {
        let bytes = tokio::fs::read(path)
            .await
            .with_context(|| format!("failed to read {}", path.display()))?;
        let mime_type = mime_guess::from_path(path)
            .first_or_octet_stream()
            .essence_str()
            .to_string();
        let display_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("document");
        let source_path = path.display().to_string();
        let metadata = UploadMetadata {
            display_name,
            mime_type: &mime_type,
            custom_metadata: vec![CustomMetadata {
                key: "source_path",
                string_value: &source_path,
            }],
        };

        logging::event(format!(
            "upload to file search store: store={store} path={} bytes={} mime_type={mime_type}",
            path.display(),
            bytes.len()
        ));
        self.upload_to_file_search_store_with_retry(store, path, &bytes, &mime_type, &metadata)
            .await
            .with_context(|| format!("failed to finalize upload for {}", path.display()))
    }

    pub async fn wait_for_operation(
        &self,
        operation: Operation,
        poll_interval: Duration,
    ) -> Result<()> {
        logging::event(format!("wait operation: operation={}", operation.name));
        let mut operation = operation;
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

            logging::debug(format!(
                "operation pending: operation={} sleeping_ms={}",
                operation.name,
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
        self.generate_content_with_optional_store(model, Some(store), prompt, system_prompt)
            .await
    }

    pub async fn generate_content_with_optional_store(
        &self,
        model: &str,
        store: Option<&str>,
        prompt: &str,
        system_prompt: Option<&str>,
    ) -> Result<GenerateContentResponse> {
        let model = model.strip_prefix("models/").unwrap_or(model);
        logging::event(format!(
            "generate content: model={model} store={} prompt_chars={} system_prompt_chars={}",
            store.unwrap_or("<none>"),
            prompt.chars().count(),
            system_prompt
                .map(str::chars)
                .map(Iterator::count)
                .unwrap_or(0)
        ));
        let url = self.url(&format!("/v1beta/models/{model}:generateContent"));
        let mut body = json!({
            "contents": [{
                "role": "user",
                "parts": [{ "text": prompt }]
            }]
        });
        if let Some(system_prompt) = system_prompt {
            body["systemInstruction"] = json!({
                "parts": [{ "text": system_prompt }]
            });
        }
        if let Some(store) = store {
            body["tools"] = json!([{
                "fileSearch": {
                    "fileSearchStoreNames": [store]
                }
            }]);
        }
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
    ) -> Result<reqwest::Response> {
        let model = model.strip_prefix("models/").unwrap_or(model);
        logging::event(format!(
            "stream generate content: model={model} store={} prompt_chars={} system_prompt_chars={}",
            store.unwrap_or("<none>"),
            prompt.chars().count(),
            system_prompt
                .map(str::chars)
                .map(Iterator::count)
                .unwrap_or(0)
        ));
        let url = self.url(&format!("/v1beta/models/{model}:streamGenerateContent"));
        let mut body = json!({
            "contents": [{
                "role": "user",
                "parts": [{ "text": prompt }]
            }]
        });
        if let Some(system_prompt) = system_prompt {
            body["systemInstruction"] = json!({
                "parts": [{ "text": system_prompt }]
            });
        }
        if let Some(store) = store {
            body["tools"] = json!([{
                "fileSearch": {
                    "fileSearchStoreNames": [store]
                }
            }]);
        }
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

    async fn start_upload(
        &self,
        store: &str,
        byte_len: usize,
        mime_type: &str,
        metadata: &UploadMetadata<'_>,
    ) -> Result<String> {
        logging::event(format!(
            "start resumable upload: store={store} bytes={byte_len} mime_type={mime_type}"
        ));
        let url = self.url(&format!("/upload/v1beta/{store}:uploadToFileSearchStore"));
        logging::debug(format!(
            "POST {url} resumable upload metadata={}",
            serde_json::to_string(metadata)
                .unwrap_or_else(|_| "<unserializable metadata>".to_string())
        ));
        let response = self
            .http
            .post(url)
            .query(&[("key", &self.api_key)])
            .header("X-Goog-Upload-Protocol", "resumable")
            .header("X-Goog-Upload-Command", "start")
            .header("X-Goog-Upload-Header-Content-Length", byte_len.to_string())
            .header("X-Goog-Upload-Header-Content-Type", mime_type)
            .json(metadata)
            .send()
            .await
            .context("failed to start resumable File Search upload")?;

        let status = response.status();
        logging::event(format!("start resumable upload response: status={status}"));
        let headers = response.headers().clone();
        logging::debug(format!(
            "start resumable upload headers: {}",
            redacted_headers(&headers)
        ));
        if !status.is_success() {
            return Err(api_error(status, response.text().await.unwrap_or_default()));
        }

        header_value(&headers, "x-goog-upload-url")
            .map(str::to_string)
            .context("Gemini upload start response did not include x-goog-upload-url")
    }

    async fn upload_to_file_search_store_with_retry(
        &self,
        store: &str,
        path: &Path,
        bytes: &[u8],
        mime_type: &str,
        metadata: &UploadMetadata<'_>,
    ) -> Result<Operation> {
        let mut backoff = UPLOAD_FINALIZE_INITIAL_BACKOFF;
        let mut last_error = None;

        for attempt in 1..=UPLOAD_FINALIZE_MAX_ATTEMPTS {
            let upload_url = self
                .start_upload(store, bytes.len(), mime_type, metadata)
                .await
                .with_context(|| {
                    format!(
                        "failed to start File Search upload for {} into {store}",
                        path.display()
                    )
                })?;
            let response = self
                .http
                .post(upload_url)
                .header(CONTENT_LENGTH, bytes.len().to_string())
                .header("X-Goog-Upload-Offset", "0")
                .header("X-Goog-Upload-Command", "upload, finalize")
                .body(bytes.to_vec())
                .send()
                .await;

            let response = match response {
                Ok(response) => response,
                Err(error) if attempt < UPLOAD_FINALIZE_MAX_ATTEMPTS => {
                    logging::event(format!(
                        "finalize upload attempt failed: path={} attempt={attempt}/{} error={error}; retrying in {}s",
                        path.display(),
                        UPLOAD_FINALIZE_MAX_ATTEMPTS,
                        backoff.as_secs()
                    ));
                    last_error = Some(anyhow!(error));
                    sleep(backoff).await;
                    backoff = next_backoff(backoff);
                    continue;
                }
                Err(error) => return Err(error).context("failed to send finalize upload request"),
            };

            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            logging::event(format!(
                "finalize upload response: path={} status={} attempt={attempt}/{}",
                path.display(),
                status,
                UPLOAD_FINALIZE_MAX_ATTEMPTS
            ));
            logging::debug(format!(
                "finalize upload response body: status={status} bytes={} body={text}",
                text.len()
            ));

            if status.is_success() {
                return serde_json::from_str(&text)
                    .with_context(|| format!("failed to parse response: {text}"));
            }

            let error = api_error(status, text);
            if attempt < UPLOAD_FINALIZE_MAX_ATTEMPTS && is_retryable_status(status) {
                logging::event(format!(
                    "finalize upload transient error: path={} attempt={attempt}/{} status={status}; retrying in {}s",
                    path.display(),
                    UPLOAD_FINALIZE_MAX_ATTEMPTS,
                    backoff.as_secs()
                ));
                last_error = Some(error);
                sleep(backoff).await;
                backoff = next_backoff(backoff);
                continue;
            }

            return Err(error);
        }

        Err(last_error.unwrap_or_else(|| anyhow!("upload finalize retry loop exhausted")))
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

impl GenerateContentResponse {
    pub fn text(&self) -> Option<String> {
        let text = self
            .candidates
            .iter()
            .filter_map(|candidate| candidate.content.as_ref())
            .flat_map(|content| &content.parts)
            .filter_map(|part| part.text.as_deref())
            .collect::<Vec<_>>()
            .join("");

        (!text.is_empty()).then_some(text)
    }
}

impl Model {
    pub fn supports_generate_content(&self) -> bool {
        self.supported_generation_methods
            .iter()
            .any(|method| method == "generateContent")
    }
}

fn header_value<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
    headers
        .get(name)
        .and_then(|value: &HeaderValue| value.to_str().ok())
}

fn redacted_headers(headers: &HeaderMap) -> String {
    let mut lines = Vec::with_capacity(headers.len());
    for (name, value) in headers {
        let value = value
            .to_str()
            .map(redact_header_value)
            .unwrap_or_else(|_| "<non-utf8>".to_string());
        lines.push(format!("{name}: {value}"));
    }
    lines.join(", ")
}

fn is_retryable_status(status: StatusCode) -> bool {
    status == StatusCode::REQUEST_TIMEOUT
        || status == StatusCode::TOO_MANY_REQUESTS
        || status.is_server_error()
}

fn next_backoff(current: Duration) -> Duration {
    current.saturating_mul(2).min(UPLOAD_FINALIZE_MAX_BACKOFF)
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
