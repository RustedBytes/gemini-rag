use std::{path::Path, time::Duration};

use anyhow::{Context, Result, bail};

use crate::{
    cli::{IngestArgs, MULTIMODAL_EMBEDDING_MODEL},
    files::collect_files,
    gemini::{FileSearchStore, GeminiClient},
    logging,
    output::print_store,
};

pub async fn ingest_folder(client: GeminiClient, args: IngestArgs) -> Result<()> {
    logging::event(format!(
        "ingest folder started: folder={}",
        args.folder.display()
    ));
    let folder = args
        .folder
        .canonicalize()
        .with_context(|| format!("folder does not exist: {}", args.folder.display()))?;
    if !folder.is_dir() {
        bail!("{} is not a directory", folder.display());
    }

    let files = collect_files(
        &folder,
        !args.no_recursive,
        args.include_hidden,
        args.max_bytes,
    )?;
    if files.is_empty() {
        bail!("no files found in {}", folder.display());
    }
    let has_images = files.iter().any(|file| is_file_search_image(file));

    let store = match args.store {
        Some(store) => {
            logging::event(format!("ingest folder using existing store: store={store}"));
            let store = client.get_store(&store).await?;
            ensure_store_supports_files(&store, has_images)?;
            logging::event(format!(
                "ingest folder store preflight ok: store={}",
                store.name
            ));
            store.name
        }
        None => {
            let display_name = args.store_display_name.unwrap_or_else(|| {
                folder
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("rag-docs")
                    .to_string()
            });
            let embedding_model = args
                .store_embedding_model
                .as_deref()
                .or(has_images.then_some(MULTIMODAL_EMBEDDING_MODEL));
            let store = client.create_store(&display_name, embedding_model).await?;
            logging::event(format!("ingest folder created store: store={}", store.name));
            println!("Created store:");
            print_store(&store);
            store.name
        }
    };

    println!("Uploading {} file(s) into {}", files.len(), store);
    logging::event(format!(
        "ingest folder upload started: store={store} file_count={} wait={} upload_batch_size={}",
        files.len(),
        !args.no_wait,
        args.upload_batch_size
    ));
    let poll_interval = Duration::from_secs(args.poll_interval_secs);

    for (batch_index, batch) in files.chunks(args.upload_batch_size).enumerate() {
        let start = batch_index * args.upload_batch_size;
        let end = start + batch.len();
        logging::event(format!(
            "ingest folder upload batch: store={store} range={}..{} total={}",
            start + 1,
            end,
            files.len()
        ));
        println!("Uploading batch {}-{} of {}", start + 1, end, files.len());
        for (offset, file) in batch.iter().enumerate() {
            logging::event(format!(
                "ingest folder file: index={} total={} path={}",
                start + offset + 1,
                files.len(),
                file.display()
            ));
            println!(
                "[{}/{}] {}",
                start + offset + 1,
                files.len(),
                file.display()
            );
        }

        let operations = client
            .upload_files_to_file_search_store(&store, batch, args.upload_batch_size)
            .await?;
        for (offset, operation) in operations.into_iter().enumerate() {
            println!(
                "  [{}/{}] operation: {}",
                start + offset + 1,
                files.len(),
                operation.name
            );
            if !args.no_wait {
                client.wait_for_operation(operation, poll_interval).await?;
                println!("  [{}/{}] indexed", start + offset + 1, files.len());
            }
        }
    }

    println!("Store ready: {store}");
    logging::event(format!("ingest folder completed: store={store}"));
    Ok(())
}

fn ensure_store_supports_files(store: &FileSearchStore, has_images: bool) -> Result<()> {
    if has_images && store.embedding_model.as_deref() != Some(MULTIMODAL_EMBEDDING_MODEL) {
        bail!(
            "{} uses embedding model {}. JPEG/PNG File Search ingestion requires {}. Create a new store with `create-store --embedding-model {}` or run `ingest` without --store so an image-capable store is created automatically.",
            store.name,
            store.embedding_model.as_deref().unwrap_or("<default>"),
            MULTIMODAL_EMBEDDING_MODEL,
            MULTIMODAL_EMBEDDING_MODEL
        );
    }

    Ok(())
}

fn is_file_search_image(path: &Path) -> bool {
    mime_guess::from_path(path)
        .first()
        .is_some_and(|mime| matches!(mime.essence_str(), "image/jpeg" | "image/png"))
}
