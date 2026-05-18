use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

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
pub(crate) struct ListStoresResponse {
    #[serde(default)]
    pub(crate) file_search_stores: Vec<FileSearchStore>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ListModelsResponse {
    #[serde(default)]
    pub(crate) models: Vec<Model>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Model {
    pub name: String,
    #[serde(default)]
    pub display_name: Option<String>,
    #[serde(default)]
    pub(crate) supported_generation_methods: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Operation {
    pub name: String,
    #[serde(default)]
    pub(crate) done: bool,
    #[serde(default)]
    pub(crate) error: Option<ApiStatus>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ApiStatus {
    #[serde(default)]
    pub(crate) code: Option<i32>,
    #[serde(default)]
    pub(crate) message: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerateContentResponse {
    #[serde(default)]
    pub candidates: Vec<Candidate>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Candidate {
    #[serde(default)]
    pub content: Option<Content>,
    #[serde(default, alias = "grounding_metadata")]
    pub grounding_metadata: Option<GroundingMetadata>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Content {
    #[serde(default)]
    pub parts: Vec<Part>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Part {
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub inline_data: Option<InlineData>,
    #[serde(default)]
    pub file_data: Option<FileData>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InlineData {
    #[serde(default)]
    pub mime_type: Option<String>,
    #[serde(default)]
    pub data: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FileData {
    #[serde(default)]
    pub mime_type: Option<String>,
    #[serde(default)]
    pub file_uri: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GroundingMetadata {
    #[serde(default, alias = "grounding_chunks")]
    pub grounding_chunks: Vec<GroundingChunk>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GroundingChunk {
    #[serde(default, alias = "retrieved_context")]
    pub retrieved_context: Option<RetrievedContext>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RetrievedContext {
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub uri: Option<String>,
    #[serde(default, alias = "file_search_store")]
    pub file_search_store: Option<String>,
    #[serde(default, alias = "custom_metadata")]
    pub custom_metadata: Vec<RetrievedCustomMetadata>,
    #[serde(default, alias = "page_number")]
    pub page_number: Option<i32>,
    #[serde(default, alias = "media_id")]
    pub media_id: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RetrievedCustomMetadata {
    pub key: String,
    #[serde(default)]
    pub string_value: Option<String>,
    #[serde(default)]
    pub numeric_value: Option<f64>,
    #[serde(default)]
    pub string_list_value: Option<StringList>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StringList {
    #[serde(default)]
    pub values: Vec<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct UploadMetadata<'a> {
    pub(crate) display_name: &'a str,
    pub(crate) mime_type: &'a str,
    pub(crate) custom_metadata: Vec<CustomMetadata<'a>>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CustomMetadata<'a> {
    pub(crate) key: &'a str,
    pub(crate) string_value: &'a str,
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

    pub fn has_non_text_parts(&self) -> bool {
        self.candidates
            .iter()
            .filter_map(|candidate| candidate.content.as_ref())
            .flat_map(|content| &content.parts)
            .any(|part| {
                part.inline_data.is_some() || part.file_data.is_some() || !part.extra.is_empty()
            })
    }
}

impl Model {
    pub fn supports_generate_content(&self) -> bool {
        self.supported_generation_methods
            .iter()
            .any(|method| method == "generateContent")
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{GenerateContentResponse, Model};

    fn response(value: serde_json::Value) -> GenerateContentResponse {
        serde_json::from_value(value).expect("valid Gemini response")
    }

    #[test]
    fn response_text_concatenates_text_parts() {
        let response = response(json!({
            "candidates": [{
                "content": {
                    "parts": [
                        { "text": "Hello" },
                        { "text": ", world" },
                        { "inlineData": { "mimeType": "image/png", "data": "abc" } }
                    ]
                }
            }]
        }));

        assert_eq!(response.text().as_deref(), Some("Hello, world"));
    }

    #[test]
    fn response_text_returns_none_when_no_text_parts_exist() {
        let response = response(json!({
            "candidates": [{
                "content": {
                    "parts": [
                        { "inlineData": { "mimeType": "image/png", "data": "abc" } }
                    ]
                }
            }]
        }));

        assert_eq!(response.text(), None);
    }

    #[test]
    fn has_non_text_parts_detects_inline_file_and_extra_parts() {
        let inline_response = response(json!({
            "candidates": [{
                "content": {
                    "parts": [
                        { "inlineData": { "mimeType": "image/png", "data": "abc" } }
                    ]
                }
            }]
        }));
        let file_response = response(json!({
            "candidates": [{
                "content": {
                    "parts": [
                        { "fileData": { "mimeType": "image/png", "fileUri": "files/image" } }
                    ]
                }
            }]
        }));
        let extra_response = response(json!({
            "candidates": [{
                "content": {
                    "parts": [
                        { "text": "hello", "thought": true }
                    ]
                }
            }]
        }));
        let text_response = response(json!({
            "candidates": [{
                "content": {
                    "parts": [
                        { "text": "hello" }
                    ]
                }
            }]
        }));

        assert!(inline_response.has_non_text_parts());
        assert!(file_response.has_non_text_parts());
        assert!(extra_response.has_non_text_parts());
        assert!(!text_response.has_non_text_parts());
    }

    #[test]
    fn model_supports_generate_content_only_when_method_is_present() {
        let supported: Model = serde_json::from_value(json!({
            "name": "models/gemini-3-flash-preview",
            "supportedGenerationMethods": ["countTokens", "generateContent"]
        }))
        .expect("supported model");
        let unsupported: Model = serde_json::from_value(json!({
            "name": "models/gemini-embedding-2",
            "supportedGenerationMethods": ["embedContent"]
        }))
        .expect("unsupported model");

        assert!(supported.supports_generate_content());
        assert!(!unsupported.supports_generate_content());
    }
}
