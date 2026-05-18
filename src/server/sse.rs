use std::{convert::Infallible, sync::Arc};

use axum::response::sse::{Event, Sse};
use futures_util::StreamExt;
use serde_json::{Value, json};

use super::{
    AppState,
    citations::{citation_count_from_responses, file_references, markdown_citations},
    gemini_images, gemini_metadata_values,
    util::{token_estimate, unix_timestamp, unix_timestamp_millis},
};
use crate::{gemini::GenerateContentResponse, logging};

pub(super) struct StreamChatCompletionInput {
    pub(super) state: Arc<AppState>,
    pub(super) model: String,
    pub(super) requested_model: String,
    pub(super) store: Option<String>,
    pub(super) store_label: String,
    pub(super) prompt: String,
    pub(super) system_prompt: Option<String>,
    pub(super) response_modalities: Vec<String>,
}

pub(super) fn stream_chat_completion(
    input: StreamChatCompletionInput,
) -> Sse<impl futures_util::Stream<Item = std::result::Result<Event, Infallible>>> {
    let StreamChatCompletionInput {
        state,
        model,
        requested_model,
        store,
        store_label,
        prompt,
        system_prompt,
        response_modalities,
    } = input;
    let id = format!("chatcmpl-{}", unix_timestamp_millis());
    let created = unix_timestamp();
    let prompt_tokens = token_estimate(&prompt);
    let stream = async_stream::stream! {
        logging::event(format!(
            "chat completion stream started: model={model} requested_model={requested_model} store={} prompt_tokens={}",
            store_label,
            prompt_tokens
        ));
        yield sse_json(openai_stream_chunk(&id, created, &model, Some("assistant"), None, None, None));

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
                &response_modalities,
            )
            .await
        {
            Ok(response) => response.bytes_stream(),
            Err(error) => {
                logging::error(format!("chat completion stream failed to start: {error:#}"));
                yield sse_json(openai_stream_error(error.to_string()));
                yield sse_done();
                return;
            }
        };

        while let Some(chunk) = upstream.next().await {
            let chunk = match chunk {
                Ok(chunk) => chunk,
                Err(error) => {
                    logging::error(format!("chat completion stream read failed: {error:#}"));
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
                        logging::error(format!("chat completion stream parse failed: {error}: {data}"));
                        yield sse_json(openai_stream_error(format!("failed to parse Gemini stream chunk: {error}")));
                        yield sse_done();
                        return;
                    }
                };
                if let Some(text) = gemini_response.text() {
                    completion_text.push_str(&text);
                    chunk_count += 1;
                    yield sse_json(openai_stream_chunk(&id, created, &model, None, Some(&text), None, None));
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
                        yield sse_json(openai_stream_chunk(&id, created, &model, None, Some(&text), None, None));
                    }
                    streamed_responses.push(gemini_response);
                }
                Err(error) => {
                    logging::error(format!("chat completion stream parse failed: {error}: {data}"));
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
            yield sse_json(openai_stream_chunk(&id, created, &model, None, Some(&citations), None, None));
        }

        let images = streamed_responses
            .iter()
            .flat_map(gemini_images)
            .collect::<Vec<_>>();
        let metadata = json!({
            "gemini": gemini_metadata_values(&streamed_responses),
            "images": images,
            "references": file_references(&streamed_responses),
        });
        yield sse_json(openai_stream_chunk(&id, created, &model, None, None, Some("stop"), Some(metadata)));
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
    metadata: Option<Value>,
) -> Value {
    let mut delta = serde_json::Map::new();
    if let Some(role) = role {
        delta.insert("role".to_string(), json!(role));
    }
    if let Some(content) = content {
        delta.insert("content".to_string(), json!(content));
    }

    let mut chunk = json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason
        }]
    });
    if let Some(metadata) = metadata {
        chunk["metadata"] = metadata;
    }

    chunk
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
