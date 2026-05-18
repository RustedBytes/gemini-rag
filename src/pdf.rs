use std::{
    ffi::OsStr,
    path::{Path, PathBuf},
    process::Command,
    time::Duration,
};

use anyhow::{Context, Result, bail};
use tempfile::TempDir;
use tokio::time::sleep;

use crate::{cli::IngestPdfArgs, gemini::GeminiClient, logging};

pub async fn ingest_pdf(client: GeminiClient, args: IngestPdfArgs) -> Result<()> {
    logging::event(format!(
        "ingest pdf started: pdf={} store={} dpi={} first_page={:?} last_page={:?} ocr_model={} upload_jpegs={} wait={} upload_batch_size={} upload_delay_secs={} operation_timeout_secs={}",
        args.pdf.display(),
        args.store,
        args.dpi,
        args.first_page,
        args.last_page,
        args.ocr_model,
        args.upload_jpegs,
        !args.no_wait,
        args.upload_batch_size,
        args.upload_delay_secs,
        args.operation_timeout_secs
    ));
    let pdf = args
        .pdf
        .canonicalize()
        .with_context(|| format!("PDF does not exist: {}", args.pdf.display()))?;
    if !pdf.is_file() {
        bail!("{} is not a file", pdf.display());
    }

    client.get_store(&args.store).await?;
    logging::event(format!(
        "ingest pdf store preflight ok: store={}",
        args.store
    ));

    let temp_dir =
        tempfile::tempdir().context("failed to create temporary PDF render directory")?;
    let pages = render_pdf_pages(&pdf, args.dpi, args.first_page, args.last_page, &temp_dir)?;
    if pages.is_empty() {
        bail!(
            "pdftoppm did not produce any JPEG pages for {}",
            pdf.display()
        );
    }

    let upload_pages = if args.upload_jpegs {
        logging::event(format!(
            "ingest pdf direct jpeg upload selected: page_count={}",
            pages.len()
        ));
        println!(
            "Uploading {} rendered JPEG page(s) from {} into {}",
            pages.len(),
            pdf.display(),
            args.store
        );
        pages
    } else {
        logging::event(format!(
            "ingest pdf OCR selected: page_count={} model={}",
            pages.len(),
            args.ocr_model
        ));
        println!(
            "Extracting text from {} rendered page(s) from {}",
            pages.len(),
            pdf.display()
        );
        ocr_pages(&client, &args.ocr_model, &pages, &temp_dir).await?
    };

    println!(
        "Uploading {} page document(s) into {}",
        upload_pages.len(),
        args.store
    );
    let poll_interval = Duration::from_secs(args.poll_interval_secs);
    let upload_delay = Duration::from_secs(args.upload_delay_secs);
    let operation_timeout =
        (args.operation_timeout_secs > 0).then(|| Duration::from_secs(args.operation_timeout_secs));

    for (batch_index, batch) in upload_pages.chunks(args.upload_batch_size).enumerate() {
        if batch_index > 0 && !upload_delay.is_zero() {
            logging::event(format!(
                "ingest pdf upload delay: store={} delay_secs={}",
                args.store,
                upload_delay.as_secs()
            ));
            sleep(upload_delay).await;
        }

        let start = batch_index * args.upload_batch_size;
        let end = start + batch.len();
        logging::event(format!(
            "ingest pdf upload page batch: store={} range={}..{} total={}",
            args.store,
            start + 1,
            end,
            upload_pages.len()
        ));
        println!(
            "Uploading batch {}-{} of {}",
            start + 1,
            end,
            upload_pages.len()
        );
        for (offset, page) in batch.iter().enumerate() {
            logging::event(format!(
                "ingest pdf upload page document: index={} total={} path={}",
                start + offset + 1,
                upload_pages.len(),
                page.display()
            ));
            println!(
                "[{}/{}] {}",
                start + offset + 1,
                upload_pages.len(),
                page.display()
            );
        }

        let operations = client
            .upload_files_to_file_search_store(&args.store, batch, args.upload_batch_size)
            .await?;
        for (offset, operation) in operations.into_iter().enumerate() {
            println!(
                "  [{}/{}] operation: {}",
                start + offset + 1,
                upload_pages.len(),
                operation.name
            );
            if !args.no_wait {
                client
                    .wait_for_operation(operation, poll_interval, operation_timeout)
                    .await?;
                println!("  [{}/{}] indexed", start + offset + 1, upload_pages.len());
            }
        }
    }

    println!("Store ready: {}", args.store);
    logging::event(format!("ingest pdf completed: store={}", args.store));
    Ok(())
}

async fn ocr_pages(
    client: &GeminiClient,
    model: &str,
    pages: &[PathBuf],
    temp_dir: &TempDir,
) -> Result<Vec<PathBuf>> {
    let mut documents = Vec::with_capacity(pages.len());
    let prompt = concat!(
        "Extract all readable text from this page image. ",
        "Preserve headings, paragraphs, lists, equations, and table text as plain text. ",
        "Return only the extracted page text."
    );

    for (index, page) in pages.iter().enumerate() {
        logging::event(format!(
            "ingest pdf OCR page: index={} total={} path={}",
            index + 1,
            pages.len(),
            page.display()
        ));
        println!("OCR [{}/{}] {}", index + 1, pages.len(), page.display());
        let text = client.extract_text_from_image(model, page, prompt).await?;
        let document = temp_dir.path().join(format!("page-{:04}.txt", index + 1));
        let body = format!(
            "Source image: {}\nPage: {}\n\n{}",
            page.file_name()
                .and_then(OsStr::to_str)
                .unwrap_or("rendered-page.jpg"),
            index + 1,
            text.trim()
        );
        tokio::fs::write(&document, body)
            .await
            .with_context(|| format!("failed to write OCR text {}", document.display()))?;
        logging::event(format!(
            "ingest pdf OCR text written: path={} chars={}",
            document.display(),
            text.chars().count()
        ));
        documents.push(document);
    }

    Ok(documents)
}

fn render_pdf_pages(
    pdf: &Path,
    dpi: u16,
    first_page: Option<u16>,
    last_page: Option<u16>,
    temp_dir: &TempDir,
) -> Result<Vec<PathBuf>> {
    logging::event(format!(
        "render pdf pages: pdf={} dpi={dpi} first_page={first_page:?} last_page={last_page:?} temp_dir={}",
        pdf.display(),
        temp_dir.path().display()
    ));
    if let (Some(first_page), Some(last_page)) = (first_page, last_page)
        && first_page > last_page
    {
        bail!("--first-page cannot be greater than --last-page");
    }

    let stem = pdf
        .file_stem()
        .and_then(OsStr::to_str)
        .unwrap_or("document");
    let output_prefix = temp_dir.path().join(format!("{stem}-page"));
    let mut command = Command::new("pdftoppm");
    command.arg("-jpeg").arg("-r").arg(dpi.to_string());
    if let Some(first_page) = first_page {
        command.arg("-f").arg(first_page.to_string());
    }
    if let Some(last_page) = last_page {
        command.arg("-l").arg(last_page.to_string());
    }
    let output = command
        .arg(pdf)
        .arg(&output_prefix)
        .output()
        .context("failed to run pdftoppm; install poppler-utils to render PDF pages")?;

    if !output.status.success() {
        logging::event(format!(
            "render pdf pages failed: status={} stderr={}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        ));
        bail!(
            "pdftoppm failed with status {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }

    let mut pages = std::fs::read_dir(temp_dir.path())
        .context("failed to read rendered PDF pages")?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<std::io::Result<Vec<_>>>()
        .context("failed to inspect rendered PDF pages")?
        .into_iter()
        .filter(|path| has_extension(path, "jpg"))
        .collect::<Vec<_>>();

    pages.sort_by_key(|path| rendered_page_number(path).unwrap_or(usize::MAX));
    logging::event(format!("render pdf pages complete: count={}", pages.len()));
    Ok(pages)
}

fn has_extension(path: &Path, expected: &str) -> bool {
    path.extension()
        .and_then(OsStr::to_str)
        .is_some_and(|extension| extension.eq_ignore_ascii_case(expected))
}

fn rendered_page_number(path: &Path) -> Option<usize> {
    path.file_stem()
        .and_then(OsStr::to_str)
        .and_then(|stem| stem.rsplit_once('-'))
        .and_then(|(_, page)| page.parse().ok())
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::{has_extension, rendered_page_number};

    #[test]
    fn has_extension_is_case_insensitive() {
        assert!(has_extension(Path::new("page-1.JPG"), "jpg"));
        assert!(has_extension(Path::new("page-1.jpg"), "JPG"));
        assert!(!has_extension(Path::new("page-1.png"), "jpg"));
        assert!(!has_extension(Path::new("page-1"), "jpg"));
    }

    #[test]
    fn rendered_page_number_reads_trailing_numeric_suffix() {
        assert_eq!(
            rendered_page_number(Path::new("document-page-1.jpg")),
            Some(1)
        );
        assert_eq!(
            rendered_page_number(Path::new("document-page-0007.jpg")),
            Some(7)
        );
        assert_eq!(
            rendered_page_number(Path::new("document-page-final.jpg")),
            None
        );
        assert_eq!(rendered_page_number(Path::new("document.jpg")), None);
    }
}
