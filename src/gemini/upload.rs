use std::{path::Path, time::Duration};

use anyhow::{Context, Result, anyhow};
use reqwest::header::CONTENT_LENGTH;
use tokio::time::sleep;

use super::{
    GeminiClient, api_error, header_value, is_retryable_status, next_backoff, redacted_headers,
};
use crate::gemini::types::{CustomMetadata, Operation, UploadMetadata};
use crate::logging;

const UPLOAD_FINALIZE_MAX_ATTEMPTS: usize = 5;
const UPLOAD_FINALIZE_INITIAL_BACKOFF: Duration = Duration::from_secs(2);
pub(super) const UPLOAD_FINALIZE_MAX_BACKOFF: Duration = Duration::from_secs(30);

impl GeminiClient {
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
}
