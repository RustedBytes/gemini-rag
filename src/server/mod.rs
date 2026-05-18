mod citations;
mod error;
mod sse;
mod types;
mod util;

use std::{net::SocketAddr, sync::Arc, time::Instant};

use anyhow::{Context, Result, anyhow, bail};
use axum::{
    Json, Router,
    body::Body,
    extract::State,
    http::{HeaderMap, HeaderValue, Request},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde_json::{Value, json};

use crate::{
    cli::ServeArgs,
    gemini::{GeminiClient, GenerateContentResponse, Model},
    logging,
};
use citations::{citation_count, file_references, with_markdown_citations};
use error::ApiError;
use sse::{StreamChatCompletionInput, stream_chat_completion};
use types::{
    AssistantMessage, ChatCompletionRequest, ChatCompletionResponse, Choice, ModelListResponse,
    ModelObject, Usage, chat_prompt,
};
use util::{normalize_model_name, token_estimate, unix_timestamp, unix_timestamp_millis};

#[derive(Clone)]
struct AppState {
    pub(super) client: GeminiClient,
    default_store: Option<String>,
    default_model: String,
    system_prompt: Option<String>,
}

pub async fn serve_openai_proxy(client: GeminiClient, args: ServeArgs) -> Result<()> {
    let bind: SocketAddr = args
        .bind
        .parse()
        .with_context(|| format!("invalid bind address: {}", args.bind))?;
    let default_model = normalize_model_name(&args.model);
    let system_prompt = read_optional_system_prompt(args.system_prompt_file.as_ref()).await?;
    logging::event(format!(
        "server configured: bind={} default_store={} default_model={} system_prompt_chars={}",
        bind,
        args.store.as_deref().unwrap_or("<none>"),
        default_model,
        system_prompt
            .as_deref()
            .map(str::chars)
            .map(Iterator::count)
            .unwrap_or(0)
    ));
    let state = Arc::new(AppState {
        client,
        default_store: args.store,
        default_model,
        system_prompt,
    });
    let app = Router::new()
        .route("/healthz", get(healthz))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/chat/completions", post(chat_completions))
        .layer(middleware::from_fn(log_request))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind(bind)
        .await
        .with_context(|| format!("failed to bind {bind}"))?;

    println!("OpenAI-compatible Gemini RAG proxy listening on http://{bind}");
    logging::event(format!("server listening: bind={bind}"));
    axum::serve(listener, app).await.context("server failed")
}

async fn healthz() -> Json<Value> {
    Json(json!({ "status": "ok" }))
}

async fn log_request(request: Request<Body>, next: Next) -> Response {
    let started = Instant::now();
    let method = request.method().clone();
    let version = request.version();
    let headers = redacted_request_headers(request.headers());
    let path = request
        .uri()
        .path_and_query()
        .map(|path| path.as_str().to_string())
        .unwrap_or_else(|| request.uri().path().to_string());

    logging::event(format!("http request started: method={method} path={path}"));
    logging::debug(format!(
        "http request detail: method={method} path={path} version={version:?} headers={headers:#?}"
    ));
    let response = next.run(request).await;
    logging::event(format!(
        "http request completed: method={method} path={path} status={} elapsed_ms={}",
        response.status().as_u16(),
        started.elapsed().as_millis()
    ));

    response
}

fn redacted_request_headers(headers: &HeaderMap) -> String {
    headers
        .iter()
        .map(|(name, value)| {
            let value = if is_sensitive_header(name.as_str()) {
                "<redacted>".to_string()
            } else {
                header_value(value)
            };
            format!("{name}: {value}")
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn is_sensitive_header(name: &str) -> bool {
    name.eq_ignore_ascii_case("authorization")
        || name.eq_ignore_ascii_case("proxy-authorization")
        || name.eq_ignore_ascii_case("x-api-key")
        || name.eq_ignore_ascii_case("api-key")
}

fn header_value(value: &HeaderValue) -> String {
    value
        .to_str()
        .map(str::to_string)
        .unwrap_or_else(|_| "<non-utf8>".to_string())
}

async fn list_models(
    State(state): State<Arc<AppState>>,
) -> Result<Json<ModelListResponse>, ApiError> {
    let models = state.client.list_models().await?;
    logging::debug(format!("proxy models fetched: count={}", models.len()));
    let created = unix_timestamp();
    Ok(Json(ModelListResponse {
        object: "list",
        data: models
            .into_iter()
            .filter(Model::supports_generate_content)
            .map(|model| ModelObject {
                id: model
                    .name
                    .strip_prefix("models/")
                    .unwrap_or(&model.name)
                    .to_string(),
                object: "model",
                created,
                owned_by: "google",
            })
            .collect(),
    }))
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
    let requested_model = request.model.as_deref().unwrap_or("<omitted>");
    let model = request
        .model
        .as_deref()
        .map(normalize_model_name)
        .unwrap_or_else(|| state.default_model.clone());
    let response_modalities = gemini_response_modalities(&request.modalities);
    let store = request
        .store
        .clone()
        .or_else(|| state.default_store.clone());
    let store_label = store.as_deref().unwrap_or("<none>").to_string();
    let prompt = match chat_prompt(&request.messages) {
        Ok(prompt) => prompt,
        Err(error) => {
            logging::warn(format!("chat completion rejected: {error}"));
            return Err(ApiError::bad_request(error.to_string()));
        }
    };

    logging::event(format!(
        "chat completion: model={model} requested_model={requested_model} store={} messages={} prompt_chars={} system_prompt_chars={}",
        store_label,
        request.messages.len(),
        prompt.chars().count(),
        state
            .system_prompt
            .as_deref()
            .map(str::chars)
            .map(Iterator::count)
            .unwrap_or(0)
    ));
    logging::debug(format!(
        "chat completion request detail: stream={} response_modalities={} prompt={prompt:?}",
        request.stream,
        response_modalities.join(",")
    ));

    if request.stream {
        let system_prompt = state.system_prompt.clone();
        return Ok(stream_chat_completion(StreamChatCompletionInput {
            state,
            model,
            requested_model: requested_model.to_string(),
            store,
            store_label,
            prompt,
            system_prompt,
            response_modalities,
        })
        .into_response());
    }

    let gemini_response = match state
        .client
        .generate_content_with_optional_store(
            &model,
            store.as_deref(),
            &prompt,
            state.system_prompt.as_deref(),
            &response_modalities,
        )
        .await
    {
        Ok(response) => response,
        Err(error) => {
            logging::error(format!("chat completion failed: {error:#}"));
            return Err(error.into());
        }
    };
    let has_non_text_parts = gemini_response.has_non_text_parts();
    let text = match gemini_response.text() {
        Some(text) => text,
        None if has_non_text_parts => String::new(),
        None => {
            let error = anyhow!("Gemini response did not include answer text");
            logging::error(format!("chat completion failed: {error:#}"));
            return Err(error.into());
        }
    };
    let content = with_markdown_citations(text, std::slice::from_ref(&gemini_response));
    let completion_tokens = token_estimate(&content);
    let prompt_tokens = token_estimate(&prompt);
    let images = gemini_images(&gemini_response);
    let references = file_references(std::slice::from_ref(&gemini_response));
    let message_metadata = (!images.is_empty()).then(|| json!({ "images": images.clone() }));
    let metadata = json!({
        "gemini": gemini_metadata_value(&gemini_response),
        "images": images,
        "references": references,
    });
    logging::event(format!(
        "chat completion succeeded: model={model} requested_model={requested_model} store={} prompt_tokens={} completion_tokens={} citation_count={}",
        store_label,
        prompt_tokens,
        completion_tokens,
        citation_count(&gemini_response)
    ));

    Ok(Json(ChatCompletionResponse {
        id: format!("chatcmpl-{}", unix_timestamp_millis()),
        object: "chat.completion",
        created: unix_timestamp(),
        model,
        choices: vec![Choice {
            index: 0,
            message: AssistantMessage {
                role: "assistant",
                content,
                metadata: message_metadata,
            },
            finish_reason: "stop",
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens.saturating_add(completion_tokens),
        },
        metadata,
    })
    .into_response())
}

fn gemini_images(response: &GenerateContentResponse) -> Vec<Value> {
    response
        .candidates
        .iter()
        .enumerate()
        .filter_map(|(candidate_index, candidate)| {
            candidate
                .content
                .as_ref()
                .map(|content| (candidate_index, content))
        })
        .flat_map(|(candidate_index, content)| {
            content
                .parts
                .iter()
                .enumerate()
                .filter_map(move |(part_index, part)| {
                    if let Some(inline_data) = &part.inline_data {
                        let mime_type = inline_data.mime_type.clone();
                        let data = inline_data.data.clone();
                        let data_url = match (mime_type.as_deref(), data.as_deref()) {
                            (Some(mime_type), Some(data)) => {
                                Some(format!("data:{mime_type};base64,{data}"))
                            }
                            _ => None,
                        };
                        return Some(json!({
                            "candidate_index": candidate_index,
                            "part_index": part_index,
                            "source": "inlineData",
                            "mime_type": mime_type,
                            "data": data,
                            "data_url": data_url,
                        }));
                    }

                    part.file_data.as_ref().map(|file_data| {
                        json!({
                            "candidate_index": candidate_index,
                            "part_index": part_index,
                            "source": "fileData",
                            "mime_type": file_data.mime_type,
                            "file_uri": file_data.file_uri,
                        })
                    })
                })
        })
        .collect()
}

pub(super) fn gemini_metadata_value(response: &GenerateContentResponse) -> Value {
    snake_case_json_keys(serde_json::to_value(response).unwrap_or(Value::Null))
}

pub(super) fn gemini_metadata_values(responses: &[GenerateContentResponse]) -> Value {
    Value::Array(responses.iter().map(gemini_metadata_value).collect())
}

fn snake_case_json_keys(value: Value) -> Value {
    match value {
        Value::Object(map) => Value::Object(
            map.into_iter()
                .map(|(key, value)| (to_snake_case(&key), snake_case_json_keys(value)))
                .collect(),
        ),
        Value::Array(values) => {
            Value::Array(values.into_iter().map(snake_case_json_keys).collect())
        }
        value => value,
    }
}

fn to_snake_case(key: &str) -> String {
    key.chars().enumerate().fold(
        String::with_capacity(key.len()),
        |mut snake, (index, character)| {
            if character.is_ascii_uppercase() {
                if index > 0 {
                    snake.push('_');
                }
                snake.push(character.to_ascii_lowercase());
            } else {
                snake.push(character);
            }
            snake
        },
    )
}

fn gemini_response_modalities(modalities: &[String]) -> Vec<String> {
    let mut response_modalities = modalities
        .iter()
        .filter_map(|modality| match modality.to_ascii_lowercase().as_str() {
            "text" => Some("TEXT".to_string()),
            "image" => Some("IMAGE".to_string()),
            _ => None,
        })
        .collect::<Vec<_>>();

    if response_modalities
        .iter()
        .any(|modality| modality == "IMAGE")
        && !response_modalities
            .iter()
            .any(|modality| modality == "TEXT")
    {
        response_modalities.insert(0, "TEXT".to_string());
    }

    response_modalities.dedup();
    response_modalities
}

async fn read_optional_system_prompt(path: Option<&std::path::PathBuf>) -> Result<Option<String>> {
    let Some(path) = path else {
        return Ok(None);
    };
    if path.as_os_str().is_empty() {
        return Ok(None);
    }

    let prompt = tokio::fs::read_to_string(path)
        .await
        .with_context(|| format!("failed to read system prompt file {}", path.display()))?;
    if prompt.trim().is_empty() {
        bail!("system prompt file is empty: {}", path.display());
    }
    logging::event(format!(
        "server system prompt loaded: path={} chars={}",
        path.display(),
        prompt.trim().chars().count()
    ));

    Ok(Some(prompt.trim().to_string()))
}

#[cfg(test)]
mod tests {
    use axum::http::{HeaderMap, HeaderValue};

    use super::redacted_request_headers;

    #[test]
    fn redacts_sensitive_request_headers() {
        let mut headers = HeaderMap::new();
        headers.insert("authorization", HeaderValue::from_static("Bearer secret"));
        headers.insert("x-api-key", HeaderValue::from_static("secret-key"));
        headers.insert("content-type", HeaderValue::from_static("application/json"));

        let formatted = redacted_request_headers(&headers);

        assert!(formatted.contains("authorization: <redacted>"));
        assert!(formatted.contains("x-api-key: <redacted>"));
        assert!(formatted.contains("content-type: application/json"));
        assert!(!formatted.contains("Bearer secret"));
        assert!(!formatted.contains("secret-key"));
    }
}
