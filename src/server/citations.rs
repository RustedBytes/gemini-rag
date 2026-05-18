use crate::gemini::GenerateContentResponse;

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

pub(super) fn citation_count(response: &GenerateContentResponse) -> usize {
    single_response_citation_count(response)
}

pub(super) fn citation_count_from_responses(responses: &[GenerateContentResponse]) -> usize {
    responses.iter().map(single_response_citation_count).sum()
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
