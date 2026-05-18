use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Deserialize)]
pub(super) struct ChatCompletionRequest {
    #[serde(default)]
    pub(super) model: Option<String>,
    pub(super) messages: Vec<ChatMessage>,
    #[serde(default)]
    pub(super) stream: bool,
    #[serde(default, alias = "file_search_store", alias = "fileSearchStore")]
    pub(super) store: Option<String>,
    #[serde(default, alias = "response_modalities", alias = "responseModalities")]
    pub(super) modalities: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub(super) struct ChatMessage {
    pub(super) role: String,
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
pub(super) struct ChatCompletionResponse {
    pub(super) id: String,
    pub(super) object: &'static str,
    pub(super) created: u64,
    pub(super) model: String,
    pub(super) choices: Vec<Choice>,
    pub(super) usage: Usage,
    pub(super) metadata: Value,
}

#[derive(Debug, Serialize)]
pub(super) struct Choice {
    pub(super) index: u32,
    pub(super) message: AssistantMessage,
    pub(super) finish_reason: &'static str,
}

#[derive(Debug, Serialize)]
pub(super) struct AssistantMessage {
    pub(super) role: &'static str,
    pub(super) content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) metadata: Option<Value>,
}

#[derive(Debug, Serialize)]
pub(super) struct Usage {
    pub(super) prompt_tokens: u32,
    pub(super) completion_tokens: u32,
    pub(super) total_tokens: u32,
}

#[derive(Debug, Serialize)]
pub(super) struct ModelListResponse {
    pub(super) object: &'static str,
    pub(super) data: Vec<ModelObject>,
}

#[derive(Debug, Serialize)]
pub(super) struct ModelObject {
    pub(super) id: String,
    pub(super) object: &'static str,
    pub(super) created: u64,
    pub(super) owned_by: &'static str,
}

pub(super) fn chat_prompt(messages: &[ChatMessage]) -> Result<String> {
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

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{ChatCompletionRequest, ChatMessage, chat_prompt};

    fn messages(value: serde_json::Value) -> Vec<ChatMessage> {
        serde_json::from_value(value).expect("valid chat messages")
    }

    #[test]
    fn chat_prompt_formats_text_and_text_parts() {
        let messages = messages(json!([
            { "role": "system", "content": "  Be concise.  " },
            {
                "role": "user",
                "content": [
                    { "type": "input_text", "text": "First line" },
                    { "type": "image_url", "image_url": { "url": "https://example.test/a.png" } },
                    { "type": "text", "text": "Second line" },
                    { "text": "Implicit text part" }
                ]
            }
        ]));

        let prompt = chat_prompt(&messages).expect("prompt");

        assert_eq!(
            prompt,
            "system: Be concise.\n\nuser: First line\nSecond line\nImplicit text part"
        );
    }

    #[test]
    fn chat_prompt_rejects_missing_text() {
        let messages = messages(json!([
            {
                "role": "user",
                "content": [
                    { "type": "image_url", "image_url": { "url": "https://example.test/a.png" } }
                ]
            }
        ]));

        let error = chat_prompt(&messages).expect_err("missing text should fail");

        assert_eq!(
            error.to_string(),
            "messages did not include any text content"
        );
        assert_eq!(
            chat_prompt(&[])
                .expect_err("empty messages should fail")
                .to_string(),
            "messages must not be empty"
        );
    }

    #[test]
    fn chat_completion_request_deserializes_store_and_modality_aliases() {
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "models/gemini-3-flash-preview",
            "messages": [{ "role": "user", "content": "Hello" }],
            "stream": true,
            "fileSearchStore": "fileSearchStores/demo",
            "response_modalities": ["text", "image"]
        }))
        .expect("request");

        assert_eq!(
            request.model.as_deref(),
            Some("models/gemini-3-flash-preview")
        );
        assert!(request.stream);
        assert_eq!(request.store.as_deref(), Some("fileSearchStores/demo"));
        assert_eq!(request.modalities, ["text", "image"]);
    }
}
