use std::{
    convert::Infallible,
    net::SocketAddr,
    sync::Arc,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, anyhow, bail};
use axum::{
    Json, Router,
    body::Body,
    extract::State,
    http::{Request, StatusCode},
    middleware::{self, Next},
    response::{
        IntoResponse, Response,
        sse::{Event, Sse},
    },
    routing::{get, post},
};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{
    cli::ServeArgs,
    gemini::{GeminiClient, GenerateContentResponse, Model},
    logging,
};

#[derive(Clone)]
struct AppState {
    client: GeminiClient,
    default_store: Option<String>,
    default_model: String,
    system_prompt: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionRequest {
    #[serde(default)]
    model: Option<String>,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    stream: bool,
    #[serde(default, alias = "file_search_store", alias = "fileSearchStore")]
    store: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatMessage {
    role: String,
    content: MessageContent,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum MessageContent {
    Text(String),
    Parts(Vec<MessagePart>),
    Other(Value),
}

#[derive(Debug, Deserialize)]
struct MessagePart {
    #[serde(default, rename = "type")]
    part_type: Option<String>,
    #[serde(default)]
    text: Option<String>,
}

#[derive(Debug, Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<Choice>,
    usage: Usage,
}

#[derive(Debug, Serialize)]
struct Choice {
    index: u32,
    message: AssistantMessage,
    finish_reason: &'static str,
}

#[derive(Debug, Serialize)]
struct AssistantMessage {
    role: &'static str,
    content: String,
}

#[derive(Debug, Serialize)]
struct Usage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

#[derive(Debug, Serialize)]
struct ModelListResponse {
    object: &'static str,
    data: Vec<ModelObject>,
}

#[derive(Debug, Serialize)]
struct ModelObject {
    id: String,
    object: &'static str,
    created: u64,
    owned_by: &'static str,
}

pub async fn serve_openai_proxy(client: GeminiClient, args: ServeArgs) -> Result<()> {
    let bind: SocketAddr = args
        .bind
        .parse()
        .with_context(|| format!("invalid bind address: {}", args.bind))?;
    let default_model = normalize_model_name(&args.model);
    let system_prompt = read_optional_system_prompt(args.system_prompt_file.as_ref()).await?;
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
    let path = request
        .uri()
        .path_and_query()
        .map(|path| path.as_str().to_string())
        .unwrap_or_else(|| request.uri().path().to_string());

    logging::event(format!("http request started: method={method} path={path}"));
    let response = next.run(request).await;
    logging::event(format!(
        "http request completed: method={method} path={path} status={} elapsed_ms={}",
        response.status().as_u16(),
        started.elapsed().as_millis()
    ));

    response
}

async fn list_models(
    State(state): State<Arc<AppState>>,
) -> Result<Json<ModelListResponse>, ApiError> {
    let models = state.client.list_models().await?;
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
    let model = state.default_model.clone();
    let store = request
        .store
        .clone()
        .or_else(|| state.default_store.clone());
    let store_label = store.as_deref().unwrap_or("<none>").to_string();
    let prompt = match chat_prompt(&request.messages) {
        Ok(prompt) => prompt,
        Err(error) => {
            logging::event(format!("chat completion rejected: {error}"));
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

    if request.stream {
        let system_prompt = state.system_prompt.clone();
        return Ok(stream_chat_completion(
            state,
            model,
            requested_model.to_string(),
            store,
            store_label,
            prompt,
            system_prompt,
        )
        .into_response());
    }

    let gemini_response = match state
        .client
        .generate_content_with_optional_store(
            &model,
            store.as_deref(),
            &prompt,
            state.system_prompt.as_deref(),
        )
        .await
    {
        Ok(response) => response,
        Err(error) => {
            logging::event(format!("chat completion failed: {error:#}"));
            return Err(error.into());
        }
    };
    let text = match gemini_response.text() {
        Some(text) => text,
        None => {
            let error = anyhow!("Gemini response did not include answer text");
            logging::event(format!("chat completion failed: {error:#}"));
            return Err(error.into());
        }
    };
    let content = with_markdown_citations(text, std::slice::from_ref(&gemini_response));
    let completion_tokens = token_estimate(&content);
    let prompt_tokens = token_estimate(&prompt);
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
            },
            finish_reason: "stop",
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens.saturating_add(completion_tokens),
        },
    })
    .into_response())
}

fn stream_chat_completion(
    state: Arc<AppState>,
    model: String,
    requested_model: String,
    store: Option<String>,
    store_label: String,
    prompt: String,
    system_prompt: Option<String>,
) -> Sse<impl futures_util::Stream<Item = std::result::Result<Event, Infallible>>> {
    let id = format!("chatcmpl-{}", unix_timestamp_millis());
    let created = unix_timestamp();
    let prompt_tokens = token_estimate(&prompt);
    let stream = async_stream::stream! {
        logging::event(format!(
            "chat completion stream started: model={model} requested_model={requested_model} store={} prompt_tokens={}",
            store_label,
            prompt_tokens
        ));
        yield sse_json(openai_stream_chunk(&id, created, &model, Some("assistant"), None, None));

        let mut completion_text = String::new();
        let mut streamed_responses = Vec::new();
        let mut buffer = String::new();
        let mut chunk_count = 0usize;
        let mut upstream = match state
            .client
            .stream_generate_content_with_optional_store(
                &model,
                store.as_deref(),
                &prompt,
                system_prompt.as_deref(),
            )
            .await
        {
            Ok(response) => response.bytes_stream(),
            Err(error) => {
                logging::event(format!("chat completion stream failed to start: {error:#}"));
                yield sse_json(openai_stream_error(error.to_string()));
                yield sse_done();
                return;
            }
        };

        while let Some(chunk) = upstream.next().await {
            let chunk = match chunk {
                Ok(chunk) => chunk,
                Err(error) => {
                    logging::event(format!("chat completion stream read failed: {error:#}"));
                    yield sse_json(openai_stream_error(error.to_string()));
                    yield sse_done();
                    return;
                }
            };
            buffer.push_str(&String::from_utf8_lossy(&chunk));
            if buffer.contains('\r') {
                buffer = buffer.replace("\r\n", "\n");
            }

            for data in drain_sse_data_events(&mut buffer) {
                if data == "[DONE]" {
                    continue;
                }
                let gemini_response = match serde_json::from_str::<GenerateContentResponse>(&data) {
                    Ok(response) => response,
                    Err(error) => {
                        logging::event(format!("chat completion stream parse failed: {error}: {data}"));
                        yield sse_json(openai_stream_error(format!("failed to parse Gemini stream chunk: {error}")));
                        yield sse_done();
                        return;
                    }
                };
                if let Some(text) = gemini_response.text() {
                    completion_text.push_str(&text);
                    chunk_count += 1;
                    yield sse_json(openai_stream_chunk(&id, created, &model, None, Some(&text), None));
                }
                streamed_responses.push(gemini_response);
            }
        }

        for data in drain_remaining_sse_data_events(&mut buffer) {
            if data == "[DONE]" {
                continue;
            }
            match serde_json::from_str::<GenerateContentResponse>(&data) {
                Ok(gemini_response) => {
                    if let Some(text) = gemini_response.text() {
                        completion_text.push_str(&text);
                        chunk_count += 1;
                        yield sse_json(openai_stream_chunk(&id, created, &model, None, Some(&text), None));
                    }
                    streamed_responses.push(gemini_response);
                }
                Err(error) => {
                    logging::event(format!("chat completion stream parse failed: {error}: {data}"));
                    yield sse_json(openai_stream_error(format!("failed to parse Gemini stream chunk: {error}")));
                    yield sse_done();
                    return;
                }
            }
        }

        let citations = markdown_citations(&streamed_responses);
        if !citations.is_empty() {
            completion_text.push_str(&citations);
            chunk_count += 1;
            yield sse_json(openai_stream_chunk(&id, created, &model, None, Some(&citations), None));
        }

        yield sse_json(openai_stream_chunk(&id, created, &model, None, None, Some("stop")));
        yield sse_done();
        logging::event(format!(
            "chat completion stream succeeded: model={model} requested_model={requested_model} store={} prompt_tokens={} completion_tokens={} citation_count={} chunks={}",
            store_label,
            prompt_tokens,
            token_estimate(&completion_text),
            citation_count_from_responses(&streamed_responses),
            chunk_count
        ));
    };

    Sse::new(stream)
}

fn chat_prompt(messages: &[ChatMessage]) -> Result<String> {
    if messages.is_empty() {
        bail!("messages must not be empty");
    }

    let prompt = messages
        .iter()
        .filter_map(|message| {
            message
                .content
                .text()
                .map(|content| format!("{}: {}", message.role, content.trim()))
        })
        .collect::<Vec<_>>()
        .join("\n\n");

    if prompt.trim().is_empty() {
        bail!("messages did not include any text content");
    }

    Ok(prompt)
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

fn with_markdown_citations(mut text: String, responses: &[GenerateContentResponse]) -> String {
    text.push_str(&markdown_citations(responses));
    text
}

fn markdown_citations(responses: &[GenerateContentResponse]) -> String {
    let citations = responses
        .iter()
        .flat_map(|response| &response.candidates)
        .filter_map(|candidate| candidate.grounding_metadata.as_ref())
        .flat_map(|metadata| &metadata.grounding_chunks)
        .filter_map(|chunk| chunk.retrieved_context.as_ref())
        .collect::<Vec<_>>();

    if citations.is_empty() {
        return String::new();
    }

    let mut text = String::new();
    text.push_str("\n\n## Citations\n");
    for (index, citation) in citations.iter().enumerate() {
        let label = citation
            .title
            .as_deref()
            .or(citation.uri.as_deref())
            .unwrap_or("retrieved context");
        match citation.uri.as_deref() {
            Some(uri) => text.push_str(&format!("\n{}. [{}]({})", index + 1, label, uri)),
            None => text.push_str(&format!("\n{}. **{}**", index + 1, label)),
        }
        if let Some(snippet) = citation
            .text
            .as_deref()
            .map(str::trim)
            .filter(|text| !text.is_empty())
        {
            text.push_str(&format!("\n   > {}", snippet.replace('\n', " ")));
        }
        text.push('\n');
    }

    text
}

fn citation_count(response: &GenerateContentResponse) -> usize {
    single_response_citation_count(response)
}

fn citation_count_from_responses(responses: &[GenerateContentResponse]) -> usize {
    responses.iter().map(single_response_citation_count).sum()
}

fn drain_sse_data_events(buffer: &mut String) -> Vec<String> {
    let mut events = Vec::new();
    while let Some(index) = buffer.find("\n\n") {
        let raw_event = buffer[..index].to_string();
        buffer.drain(..index + 2);
        if let Some(data) = sse_event_data(&raw_event) {
            events.push(data);
        }
    }
    events
}

fn drain_remaining_sse_data_events(buffer: &mut String) -> Vec<String> {
    if buffer.trim().is_empty() {
        buffer.clear();
        return Vec::new();
    }

    let raw_event = std::mem::take(buffer);
    sse_event_data(&raw_event).into_iter().collect()
}

fn sse_event_data(raw_event: &str) -> Option<String> {
    let data = raw_event
        .lines()
        .filter_map(|line| {
            line.strip_prefix("data:")
                .map(str::trim_start)
                .map(str::to_string)
        })
        .collect::<Vec<_>>()
        .join("\n");

    (!data.is_empty()).then_some(data)
}

fn sse_json(value: Value) -> std::result::Result<Event, Infallible> {
    Ok(Event::default().data(value.to_string()))
}

fn sse_done() -> std::result::Result<Event, Infallible> {
    Ok(Event::default().data("[DONE]"))
}

fn openai_stream_chunk(
    id: &str,
    created: u64,
    model: &str,
    role: Option<&str>,
    content: Option<&str>,
    finish_reason: Option<&str>,
) -> Value {
    let mut delta = serde_json::Map::new();
    if let Some(role) = role {
        delta.insert("role".to_string(), json!(role));
    }
    if let Some(content) = content {
        delta.insert("content".to_string(), json!(content));
    }

    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason
        }]
    })
}

fn openai_stream_error(message: String) -> Value {
    json!({
        "error": {
            "message": message,
            "type": "server_error",
            "code": null
        }
    })
}

fn single_response_citation_count(response: &GenerateContentResponse) -> usize {
    response
        .candidates
        .iter()
        .filter_map(|candidate| candidate.grounding_metadata.as_ref())
        .flat_map(|metadata| &metadata.grounding_chunks)
        .filter(|chunk| chunk.retrieved_context.is_some())
        .count()
}

impl MessageContent {
    fn text(&self) -> Option<String> {
        match self {
            Self::Text(text) => Some(text.clone()),
            Self::Parts(parts) => {
                let text = parts
                    .iter()
                    .filter(|part| {
                        part.part_type
                            .as_deref()
                            .map(|part_type| part_type == "text" || part_type == "input_text")
                            .unwrap_or(true)
                    })
                    .filter_map(|part| part.text.as_deref())
                    .collect::<Vec<_>>()
                    .join("\n");

                (!text.is_empty()).then_some(text)
            }
            Self::Other(value) => value.as_str().map(str::to_string),
        }
    }
}

fn normalize_model_name(model: &str) -> String {
    match model.strip_prefix("models/").unwrap_or(model) {
        "gemini-flash-3-preview" => "gemini-3-flash-preview".to_string(),
        model => model.to_string(),
    }
}

fn token_estimate(text: &str) -> u32 {
    text.split_whitespace()
        .count()
        .try_into()
        .unwrap_or(u32::MAX)
}

fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default()
}

fn unix_timestamp_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or_default()
}

struct ApiError {
    status: StatusCode,
    message: String,
    error_type: &'static str,
}

impl ApiError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
            error_type: "invalid_request_error",
        }
    }
}

impl From<anyhow::Error> for ApiError {
    fn from(error: anyhow::Error) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: error.to_string(),
            error_type: "server_error",
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(json!({
                "error": {
                    "message": self.message,
                    "type": self.error_type,
                    "code": null
                }
            })),
        )
            .into_response()
    }
}
