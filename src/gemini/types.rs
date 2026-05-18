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
