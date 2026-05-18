use std::time::Duration;

use anyhow::{Context, Result, bail};

use crate::{
    cli::IngestArgs, files::collect_files, gemini::GeminiClient, logging, output::print_store,
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

    let store = match args.store {
        Some(store) => {
            logging::event(format!("ingest folder using existing store: store={store}"));
            store
        }
        None => {
            let display_name = args.store_display_name.unwrap_or_else(|| {
                folder
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("rag-docs")
                    .to_string()
            });
            let store = client.create_store(&display_name).await?;
            logging::event(format!("ingest folder created store: store={}", store.name));
            println!("Created store:");
            print_store(&store);
            store.name
        }
    };

    let files = collect_files(
        &folder,
        !args.no_recursive,
        args.include_hidden,
        args.max_bytes,
    )?;
    if files.is_empty() {
        bail!("no files found in {}", folder.display());
    }

    println!("Uploading {} file(s) into {}", files.len(), store);
    logging::event(format!(
        "ingest folder upload started: store={store} file_count={} wait={}",
        files.len(),
        !args.no_wait
    ));
    let poll_interval = Duration::from_secs(args.poll_interval_secs);

    for (index, file) in files.iter().enumerate() {
        logging::event(format!(
            "ingest folder file: index={} total={} path={}",
            index + 1,
            files.len(),
            file.display()
        ));
        println!("[{}/{}] {}", index + 1, files.len(), file.display());
        let operation = client.upload_to_file_search_store(&store, file).await?;
        println!("  operation: {}", operation.name);
        if !args.no_wait {
            client.wait_for_operation(operation, poll_interval).await?;
            println!("  indexed");
        }
    }

    println!("Store ready: {store}");
    logging::event(format!("ingest folder completed: store={store}"));
    Ok(())
}
