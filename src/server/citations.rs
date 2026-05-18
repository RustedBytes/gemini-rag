use crate::gemini::GenerateContentResponse;
use serde_json::{Value, json};

pub(super) fn with_markdown_citations(
    mut text: String,
    responses: &[GenerateContentResponse],
) -> String {
    text.push_str(&markdown_citations(responses));
    text
}

pub(super) fn markdown_citations(responses: &[GenerateContentResponse]) -> String {
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

    let entries = citations
        .iter()
        .enumerate()
        .map(|(index, citation)| {
            let label = citation
                .title
                .as_deref()
                .or(citation.uri.as_deref())
                .unwrap_or("retrieved context");
            let heading = citation
                .uri
                .as_deref()
                .map(|uri| format!("\n{}. [{}]({})", index + 1, label, uri))
                .unwrap_or_else(|| format!("\n{}. **{}**", index + 1, label));
            let snippet = citation
                .text
                .as_deref()
                .map(str::trim)
                .filter(|text| !text.is_empty())
                .map(|snippet| format!("\n   > {}", snippet.replace('\n', " ")))
                .unwrap_or_default();

            format!("{heading}{snippet}\n")
        })
        .collect::<Vec<_>>()
        .join("");

    format!("\n\n## Citations\n{entries}")
}

pub(super) fn citation_count(response: &GenerateContentResponse) -> usize {
    single_response_citation_count(response)
}

pub(super) fn citation_count_from_responses(responses: &[GenerateContentResponse]) -> usize {
    responses.iter().map(single_response_citation_count).sum()
}

pub(super) fn file_references(responses: &[GenerateContentResponse]) -> Vec<Value> {
    responses
        .iter()
        .flat_map(|response| &response.candidates)
        .enumerate()
        .flat_map(|(candidate_index, candidate)| {
            candidate
                .grounding_metadata
                .as_ref()
                .into_iter()
                .flat_map(move |metadata| {
                    metadata.grounding_chunks.iter().enumerate().filter_map(
                        move |(chunk_index, chunk)| {
                            let context = chunk.retrieved_context.as_ref()?;
                            let title = context.title.clone();
                            let uri = context.uri.clone();
                            let source_path = extra_string(&context.extra, "source_path")
                                .or_else(|| extra_string(&context.extra, "sourcePath"))
                                .or_else(|| extra_string(&chunk.extra, "source_path"))
                                .or_else(|| extra_string(&chunk.extra, "sourcePath"))
                                .or_else(|| uri.clone());
                            let mime_type = extra_string(&context.extra, "mime_type")
                                .or_else(|| extra_string(&context.extra, "mimeType"))
                                .or_else(|| extra_string(&chunk.extra, "mime_type"))
                                .or_else(|| extra_string(&chunk.extra, "mimeType"))
                                .or_else(|| infer_mime_type(source_path.as_deref()))
                                .or_else(|| infer_mime_type(title.as_deref()));

                            Some(json!({
                                "candidate_index": candidate_index,
                                "chunk_index": chunk_index,
                                "title": title,
                                "uri": uri,
                                "source_path": source_path,
                                "mime_type": mime_type,
                                "is_image": is_image_reference(mime_type.as_deref(), source_path.as_deref(), context.title.as_deref()),
                                "text": context.text,
                            }))
                        },
                    )
                })
        })
        .collect()
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

fn extra_string(extra: &serde_json::Map<String, Value>, key: &str) -> Option<String> {
    extra
        .get(key)
        .and_then(value_string)
        .or_else(|| extra.values().find_map(|value| nested_string(value, key)))
}

fn nested_string(value: &Value, key: &str) -> Option<String> {
    match value {
        Value::Object(map) => map
            .get(key)
            .and_then(value_string)
            .or_else(|| map.values().find_map(|value| nested_string(value, key))),
        Value::Array(values) => values.iter().find_map(|value| nested_string(value, key)),
        _ => None,
    }
}

fn value_string(value: &Value) -> Option<String> {
    match value {
        Value::String(value) => Some(value.clone()),
        Value::Object(map) => map
            .get("stringValue")
            .and_then(value_string)
            .or_else(|| map.get("string_value").and_then(value_string))
            .or_else(|| map.get("value").and_then(value_string)),
        _ => None,
    }
}

fn infer_mime_type(path: Option<&str>) -> Option<String> {
    let extension = path?
        .rsplit(['.', '?', '#'])
        .find(|part| !part.is_empty())?
        .to_ascii_lowercase();
    let mime_type = match extension.as_str() {
        "apng" => "image/apng",
        "avif" => "image/avif",
        "bmp" => "image/bmp",
        "gif" => "image/gif",
        "jpg" | "jpeg" => "image/jpeg",
        "png" => "image/png",
        "svg" => "image/svg+xml",
        "webp" => "image/webp",
        _ => return None,
    };

    Some(mime_type.to_string())
}

fn is_image_reference(
    mime_type: Option<&str>,
    source_path: Option<&str>,
    title: Option<&str>,
) -> bool {
    mime_type.is_some_and(|mime_type| mime_type.starts_with("image/"))
        || infer_mime_type(source_path).is_some()
        || infer_mime_type(title).is_some()
}
