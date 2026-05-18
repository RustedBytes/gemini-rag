mod cli;
mod files;
mod gemini;
mod ingest;
mod logging;
mod output;
mod pdf;
mod query;
mod server;

use anyhow::Result;
use clap::Parser;
use cli::{Cli, Commands};
use gemini::GeminiClient;
use ingest::ingest_folder;
use output::{print_model, print_store};
use pdf::ingest_pdf;
use query::query_store;
use server::serve_openai_proxy;

#[tokio::main]
async fn main() -> Result<()> {
    if let Err(error) = run().await {
        logging::error(format!("command failed: {error:#}"));
        return Err(error);
    }

    Ok(())
}

async fn run() -> Result<()> {
    let dotenv_loaded = dotenvy::dotenv().ok();
    let cli = Cli::parse();
    logging::init(&cli.log_file)?;
    logging::event(format!("command started: {}", cli.command.name()));
    logging::debug(format!(
        "dotenv loaded: path={}",
        dotenv_loaded
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "<none>".to_string())
    ));
    logging::debug(format!(
        "parsed cli: command={} base_url={} log_file={} api_key_present={}",
        cli.command.name(),
        cli.base_url,
        cli.log_file.display(),
        cli.api_key
            .as_deref()
            .is_some_and(|key| !key.trim().is_empty())
    ));
    let client = GeminiClient::new(cli.api_key, cli.base_url)?;

    match cli.command {
        Commands::CreateStore(args) => {
            let store = client.create_store(&args.display_name).await?;
            print_store(&store);
        }
        Commands::Ingest(args) => ingest_folder(client, args).await?,
        Commands::IngestPdf(args) => ingest_pdf(client, args).await?,
        Commands::Query(args) => query_store(client, args).await?,
        Commands::ListStores => {
            let stores = client.list_stores().await?;
            for store in stores {
                print_store(&store);
            }
        }
        Commands::ListModels => {
            let models = client.list_models().await?;
            for model in models
                .iter()
                .filter(|model| model.supports_generate_content())
            {
                print_model(model);
            }
        }
        Commands::DeleteStore(args) => {
            client.delete_store(&args.store, args.force).await?;
            println!("Deleted {}", args.store);
        }
        Commands::Serve(args) => serve_openai_proxy(client, args).await?,
    }

    logging::event("command completed");
    Ok(())
}
