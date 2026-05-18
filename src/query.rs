use anyhow::{Context, Result, anyhow, bail};

use crate::{
    cli::QueryArgs,
    gemini::GeminiClient,
    logging,
    output::{print_citations, response_text},
};

pub async fn query_store(client: GeminiClient, args: QueryArgs) -> Result<()> {
    logging::event(format!(
        "query started: store={} model={} prompt_arg_count={} show_citations={}",
        args.store,
        args.model,
        args.prompt.len(),
        args.show_citations
    ));
    let prompt = if args.prompt.is_empty() {
        logging::event("query reading prompt from stdin");
        read_stdin().await?
    } else {
        args.prompt.join(" ")
    };
    if prompt.trim().is_empty() {
        bail!("query prompt is empty");
    }
    let system_prompt = read_optional_system_prompt(args.system_prompt_file.as_ref()).await?;

    let model = normalize_model_name(&args.model);
    if model != args.model {
        logging::event(format!("query model alias: from={} to={model}", args.model));
        eprintln!("Using model alias: {} -> {}", args.model, model);
    }

    logging::event(format!(
        "query dispatch: store={} model={model} prompt_chars={} system_prompt_chars={}",
        args.store,
        prompt.trim().chars().count(),
        system_prompt
            .as_deref()
            .map(str::chars)
            .map(Iterator::count)
            .unwrap_or(0)
    ));
    let response = client
        .generate_content(&model, &args.store, prompt.trim(), system_prompt.as_deref())
        .await?;
    let text = response_text(&response)
        .ok_or_else(|| anyhow!("Gemini response did not include answer text"))?;

    println!("{text}");
    logging::event(format!(
        "query response printed: chars={}",
        text.chars().count()
    ));

    if args.show_citations {
        print_citations(&response);
        logging::event("query citations printed");
    }

    logging::event("query completed");
    Ok(())
}

async fn read_optional_system_prompt(path: Option<&std::path::PathBuf>) -> Result<Option<String>> {
    let Some(path) = path else {
        return Ok(None);
    };
    if path.as_os_str().is_empty() {
        return Ok(None);
    }

    let prompt = tokio::fs::read_to_string(path)
        .await
        .with_context(|| format!("failed to read system prompt file {}", path.display()))?;
    if prompt.trim().is_empty() {
        bail!("system prompt file is empty: {}", path.display());
    }
    logging::event(format!(
        "query system prompt loaded: path={} chars={}",
        path.display(),
        prompt.trim().chars().count()
    ));

    Ok(Some(prompt.trim().to_string()))
}

async fn read_stdin() -> Result<String> {
    use tokio::io::{AsyncReadExt, stdin};

    let mut input = String::new();
    stdin()
        .read_to_string(&mut input)
        .await
        .context("failed to read prompt from stdin")?;
    Ok(input)
}

fn normalize_model_name(model: &str) -> String {
    match model.strip_prefix("models/").unwrap_or(model) {
        "gemini-flash-3-preview" => "gemini-3-flash-preview".to_string(),
        model => model.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use anyhow::Result;

    use super::{normalize_model_name, read_optional_system_prompt};

    #[test]
    fn normalize_model_name_strips_prefix_and_handles_alias() {
        assert_eq!(
            normalize_model_name("models/gemini-flash-3-preview"),
            "gemini-3-flash-preview"
        );
        assert_eq!(
            normalize_model_name("gemini-flash-3-preview"),
            "gemini-3-flash-preview"
        );
        assert_eq!(
            normalize_model_name("models/gemini-3-flash-preview"),
            "gemini-3-flash-preview"
        );
    }

    #[tokio::test]
    async fn read_optional_system_prompt_trims_non_empty_file() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let path = temp.path().join("system-prompt.txt");
        tokio::fs::write(&path, "\n  Be precise.  \n").await?;

        let prompt = read_optional_system_prompt(Some(&path)).await?;

        assert_eq!(prompt.as_deref(), Some("Be precise."));
        assert_eq!(read_optional_system_prompt(None).await?, None);
        assert_eq!(
            read_optional_system_prompt(Some(&PathBuf::new())).await?,
            None
        );

        Ok(())
    }

    #[tokio::test]
    async fn read_optional_system_prompt_rejects_empty_file() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let path = temp.path().join("empty.txt");
        tokio::fs::write(&path, " \n\t").await?;

        let error = read_optional_system_prompt(Some(&path))
            .await
            .expect_err("empty prompt should fail");

        assert!(error.to_string().contains("system prompt file is empty"));
        Ok(())
    }
}
