use crate::gemini::{GenerateContentResponse, RetrievedContext};
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
                .or(citation.file_search_store.as_deref())
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

pub(crate) fn file_references(responses: &[GenerateContentResponse]) -> Vec<Value> {
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
                            let source_path = custom_metadata_string(context, "source_path")
                                .or_else(|| custom_metadata_string(context, "sourcePath"))
                                .or_else(|| extra_string(&context.extra, "source_path"))
                                .or_else(|| extra_string(&context.extra, "sourcePath"))
                                .or_else(|| extra_string(&chunk.extra, "source_path"))
                                .or_else(|| extra_string(&chunk.extra, "sourcePath"))
                                .or_else(|| uri.clone());
                            let mime_type = custom_metadata_string(context, "mime_type")
                                .or_else(|| custom_metadata_string(context, "mimeType"))
                                .or_else(|| extra_string(&context.extra, "mime_type"))
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
                                "file_search_store": context.file_search_store,
                                "page_number": context.page_number,
                                "media_id": context.media_id,
                                "custom_metadata": context.custom_metadata,
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

fn custom_metadata_string(context: &RetrievedContext, key: &str) -> Option<String> {
    context
        .custom_metadata
        .iter()
        .find(|metadata| metadata.key == key)
        .and_then(|metadata| {
            metadata
                .string_value
                .clone()
                .or_else(|| metadata.numeric_value.map(|value| value.to_string()))
                .or_else(|| {
                    metadata
                        .string_list_value
                        .as_ref()
                        .map(|list| list.values.join(","))
                })
        })
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

#[cfg(test)]
mod tests {
    use crate::gemini::GenerateContentResponse;
    use serde_json::json;

    use super::{citation_count, file_references, markdown_citations};

    fn response(value: serde_json::Value) -> GenerateContentResponse {
        serde_json::from_value(value).expect("valid Gemini response")
    }

    #[test]
    fn markdown_citations_formats_links_and_snippets() {
        let response = response(json!({
            "candidates": [{
                "groundingMetadata": {
                    "groundingChunks": [{
                        "retrievedContext": {
                            "title": "Setup Guide",
                            "uri": "https://example.test/setup",
                            "text": "First line\nSecond line"
                        }
                    }]
                }
            }]
        }));

        assert_eq!(citation_count(&response), 1);
        assert_eq!(
            markdown_citations(&[response]),
            "\n\n## Citations\n\n1. [Setup Guide](https://example.test/setup)\n   > First line Second line\n"
        );
    }

    #[test]
    fn file_references_extract_metadata_and_identify_images() {
        let response = response(json!({
            "candidates": [{
                "groundingMetadata": {
                    "groundingChunks": [{
                        "retrievedContext": {
                            "title": "Page image",
                            "uri": "https://example.test/page",
                            "fileSearchStore": "fileSearchStores/store-1",
                            "pageNumber": 7,
                            "mediaId": "media-1",
                            "customMetadata": [
                                { "key": "source_path", "stringValue": "docs/page-7.png" },
                                { "key": "mime_type", "stringValue": "image/png" }
                            ],
                            "text": "Page text"
                        }
                    }]
                }
            }]
        }));

        let references = file_references(&[response]);

        assert_eq!(references.len(), 1);
        let reference = &references[0];
        assert_eq!(reference["candidate_index"], 0);
        assert_eq!(reference["chunk_index"], 0);
        assert_eq!(reference["source_path"], "docs/page-7.png");
        assert_eq!(reference["mime_type"], "image/png");
        assert_eq!(reference["file_search_store"], "fileSearchStores/store-1");
        assert_eq!(reference["page_number"], 7);
        assert_eq!(reference["media_id"], "media-1");
        assert_eq!(reference["text"], "Page text");
        assert_eq!(reference["is_image"], true);
    }

    #[test]
    fn file_references_fall_back_to_nested_extra_metadata() {
        let response = response(json!({
            "candidates": [{
                "grounding_metadata": {
                    "grounding_chunks": [{
                        "retrieved_context": {
                            "title": "Notes",
                            "metadata": {
                                "document": {
                                    "sourcePath": "docs/notes.txt",
                                    "mimeType": { "stringValue": "text/plain" }
                                }
                            }
                        }
                    }]
                }
            }]
        }));

        let references = file_references(&[response]);

        assert_eq!(references.len(), 1);
        assert_eq!(references[0]["source_path"], "docs/notes.txt");
        assert_eq!(references[0]["mime_type"], "text/plain");
        assert_eq!(references[0]["is_image"], false);
    }
}
