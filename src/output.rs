use crate::gemini::{FileSearchStore, GenerateContentResponse, Model};

pub fn response_text(response: &GenerateContentResponse) -> Option<String> {
    response.text()
}

pub fn print_citations(response: &GenerateContentResponse) {
    let chunks = response
        .candidates
        .iter()
        .filter_map(|candidate| candidate.grounding_metadata.as_ref())
        .flat_map(|metadata| &metadata.grounding_chunks)
        .filter_map(|chunk| chunk.retrieved_context.as_ref())
        .collect::<Vec<_>>();

    if chunks.is_empty() {
        return;
    }

    println!("\nCitations:");
    chunks.iter().enumerate().for_each(|(index, chunk)| {
        let label = chunk
            .title
            .as_deref()
            .or(chunk.uri.as_deref())
            .unwrap_or("retrieved context");
        println!("{}. {}", index + 1, label);
        if let Some(text) = chunk.text.as_deref() {
            println!("   {}", text.replace('\n', " "));
        }
    });
}

pub fn print_store(store: &FileSearchStore) {
    match (&store.display_name, &store.embedding_model) {
        (Some(display_name), Some(embedding_model)) => {
            println!("{}\t{}\t{}", store.name, display_name, embedding_model)
        }
        (Some(display_name), None) => println!("{}\t{}", store.name, display_name),
        (None, Some(embedding_model)) => println!("{}\t{}", store.name, embedding_model),
        (None, None) => println!("{}", store.name),
    }
}

pub fn print_model(model: &Model) {
    let name = model.name.strip_prefix("models/").unwrap_or(&model.name);
    match &model.display_name {
        Some(display_name) => println!("{name}\t{display_name}"),
        None => println!("{name}"),
    }
}
