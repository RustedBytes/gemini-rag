use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};

pub const DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com";
pub const DEFAULT_MODEL: &str = "gemini-3-flash-preview";
pub const MULTIMODAL_EMBEDDING_MODEL: &str = "models/gemini-embedding-2";
pub const DEFAULT_PROXY_BIND: &str = "127.0.0.1:8080";

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Feed local files into Gemini File Search and query them."
)]
pub struct Cli {
    #[arg(long, env = "GEMINI_API_KEY", global = true)]
    pub api_key: Option<String>,

    #[arg(long, env = "GEMINI_BASE_URL", default_value = DEFAULT_BASE_URL, global = true)]
    pub base_url: String,

    #[arg(
        long,
        env = "GEMINI_RAG_LOG",
        default_value = "gemini-rag.log",
        global = true
    )]
    pub log_file: PathBuf,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Create an empty Gemini File Search store.
    CreateStore(CreateStoreArgs),
    /// Upload every file from a folder into a File Search store.
    Ingest(IngestArgs),
    /// Render each PDF page to JPEG and upload pages into a File Search store.
    IngestPdf(IngestPdfArgs),
    /// Ask a model a question grounded in a File Search store.
    Query(QueryArgs),
    /// List File Search stores in the current Gemini project.
    ListStores,
    /// List Gemini models that support generateContent.
    ListModels,
    /// Delete a File Search store.
    DeleteStore(DeleteStoreArgs),
    /// Serve an OpenAI-compatible chat completions API backed by Gemini File Search.
    Serve(ServeArgs),
}

impl Commands {
    pub fn name(&self) -> &'static str {
        match self {
            Self::CreateStore(_) => "create-store",
            Self::Ingest(_) => "ingest",
            Self::IngestPdf(_) => "ingest-pdf",
            Self::Query(_) => "query",
            Self::ListStores => "list-stores",
            Self::ListModels => "list-models",
            Self::DeleteStore(_) => "delete-store",
            Self::Serve(_) => "serve",
        }
    }
}

#[derive(Args, Debug)]
pub struct CreateStoreArgs {
    #[arg(long, default_value = "rag-docs")]
    pub display_name: String,

    /// Embedding model for the store. Use models/gemini-embedding-2 for image search.
    #[arg(long)]
    pub embedding_model: Option<String>,
}

#[derive(Args, Debug)]
pub struct IngestArgs {
    /// Folder containing files to upload.
    pub folder: PathBuf,

    /// Existing store name, for example fileSearchStores/my-docs-abc123.
    #[arg(long, env = "GEMINI_FILE_SEARCH_STORE")]
    pub store: Option<String>,

    /// Display name used when creating a store if --store is omitted.
    #[arg(long)]
    pub store_display_name: Option<String>,

    /// Embedding model used when creating a store if --store is omitted.
    #[arg(long)]
    pub store_embedding_model: Option<String>,

    /// Only upload files directly inside the folder.
    #[arg(long, default_value_t = false)]
    pub no_recursive: bool,

    /// Include hidden files and files inside hidden directories.
    #[arg(long, default_value_t = false)]
    pub include_hidden: bool,

    /// Do not wait for upload/indexing operations to finish.
    #[arg(long, default_value_t = false)]
    pub no_wait: bool,

    /// Number of files to upload concurrently.
    #[arg(long, default_value_t = 1, value_parser = parse_upload_batch_size)]
    pub upload_batch_size: usize,

    /// Seconds between operation polls.
    #[arg(long, default_value_t = 5)]
    pub poll_interval_secs: u64,

    /// Maximum seconds to wait for each upload operation. Use 0 to wait indefinitely.
    #[arg(long, default_value_t = 600)]
    pub operation_timeout_secs: u64,

    /// Skip files larger than this many bytes.
    #[arg(long)]
    pub max_bytes: Option<u64>,
}

#[derive(Args, Debug)]
pub struct IngestPdfArgs {
    /// PDF file to render and upload page-by-page.
    pub pdf: PathBuf,

    /// Existing store name, for example fileSearchStores/my-docs-abc123.
    #[arg(long, env = "GEMINI_FILE_SEARCH_STORE")]
    pub store: String,

    /// Render DPI passed to pdftoppm.
    #[arg(long, default_value_t = 200)]
    pub dpi: u16,

    /// First PDF page to render, using 1-based page numbers.
    #[arg(long)]
    pub first_page: Option<u16>,

    /// Last PDF page to render, using 1-based page numbers.
    #[arg(long)]
    pub last_page: Option<u16>,

    /// Gemini model used to extract page text from rendered JPEGs.
    #[arg(long, default_value = DEFAULT_MODEL)]
    pub ocr_model: String,

    /// Upload rendered JPEG files directly instead of OCR text pages.
    #[arg(long, default_value_t = false)]
    pub upload_jpegs: bool,

    /// Do not wait for upload/indexing operations to finish.
    #[arg(long, default_value_t = false)]
    pub no_wait: bool,

    /// Number of page documents to upload concurrently.
    #[arg(long, default_value_t = 1, value_parser = parse_upload_batch_size)]
    pub upload_batch_size: usize,

    /// Seconds between operation polls.
    #[arg(long, default_value_t = 5)]
    pub poll_interval_secs: u64,

    /// Maximum seconds to wait for each upload operation. Use 0 to wait indefinitely.
    #[arg(long, default_value_t = 600)]
    pub operation_timeout_secs: u64,
}

#[derive(Args, Debug)]
pub struct QueryArgs {
    /// Store name to ground the query with, for example fileSearchStores/my-docs-abc123.
    #[arg(long, env = "GEMINI_FILE_SEARCH_STORE")]
    pub store: String,

    /// Gemini model to use for the grounded response.
    #[arg(long, default_value = DEFAULT_MODEL)]
    pub model: String,

    /// File containing the system prompt sent as Gemini systemInstruction.
    #[arg(long, env = "GEMINI_SYSTEM_PROMPT_FILE")]
    pub system_prompt_file: Option<PathBuf>,

    /// Question to ask. If omitted, stdin is used.
    pub prompt: Vec<String>,

    /// Print retrieved citation chunks after the answer.
    #[arg(long, default_value_t = false)]
    pub show_citations: bool,
}

#[derive(Args, Debug)]
pub struct DeleteStoreArgs {
    #[arg(long, env = "GEMINI_FILE_SEARCH_STORE")]
    pub store: String,

    /// Delete documents and related objects inside the store too.
    #[arg(long, default_value_t = false)]
    pub force: bool,
}

#[derive(Args, Debug)]
pub struct ServeArgs {
    /// Socket address for the Axum server.
    #[arg(long, env = "GEMINI_RAG_BIND", default_value = DEFAULT_PROXY_BIND)]
    pub bind: String,

    /// Default File Search store used for chat completions.
    #[arg(long, env = "GEMINI_FILE_SEARCH_STORE")]
    pub store: Option<String>,

    /// Gemini model used by the proxy server.
    #[arg(long, env = "GEMINI_PROXY_MODEL", default_value = DEFAULT_MODEL)]
    pub model: String,

    /// File containing the system prompt sent as Gemini systemInstruction.
    #[arg(long, env = "GEMINI_SYSTEM_PROMPT_FILE")]
    pub system_prompt_file: Option<PathBuf>,
}

fn parse_upload_batch_size(value: &str) -> Result<usize, String> {
    let batch_size = value
        .parse::<usize>()
        .map_err(|error| format!("invalid upload batch size: {error}"))?;
    if batch_size == 0 {
        return Err("upload batch size must be at least 1".to_string());
    }

    Ok(batch_size)
}
