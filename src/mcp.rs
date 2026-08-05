use std::{net::SocketAddr, path::Path, sync::Arc};

use anyhow::{Context, Result, bail};
use axum::{
    Router,
    body::Body,
    extract::State,
    http::{Request, StatusCode, header},
    middleware::{self, Next},
    response::{IntoResponse, Response},
};
use rmcp::{
    Json, ServerHandler,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{Implementation, ServerCapabilities, ServerInfo},
    schemars::JsonSchema,
    tool, tool_handler, tool_router,
    transport::streamable_http_server::{
        StreamableHttpServerConfig, StreamableHttpService, session::local::LocalSessionManager,
    },
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    cli::McpArgs,
    gemini::{GeminiClient, GenerateContentResponse},
    logging,
    server::citations::file_references,
};

#[derive(Clone)]
struct GeminiMcpServer {
    client: GeminiClient,
    default_store: Option<String>,
    default_model: String,
    system_prompt: Option<String>,
    tool_router: ToolRouter<Self>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct QueryInput {
    /// Question or instruction to answer using Gemini File Search.
    prompt: String,
    /// File Search store name. Uses the server default when omitted.
    store: Option<String>,
    /// Gemini model name. Uses the server default when omitted.
    model: Option<String>,
}

#[derive(Debug, Serialize, JsonSchema)]
struct StoreSummary {
    name: String,
    display_name: Option<String>,
    embedding_model: Option<String>,
}

#[derive(Debug, Serialize, JsonSchema)]
struct ListStoresOutput {
    stores: Vec<StoreSummary>,
}

#[derive(Debug, Serialize, JsonSchema)]
struct ModelSummary {
    name: String,
    display_name: Option<String>,
}

#[derive(Debug, Serialize, JsonSchema)]
struct ListModelsOutput {
    models: Vec<ModelSummary>,
}

#[derive(Debug, Serialize, JsonSchema)]
struct QueryOutput {
    answer: String,
    store: String,
    model: String,
    references: Vec<Value>,
    usage: Option<Value>,
}

impl GeminiMcpServer {
    fn new(
        client: GeminiClient,
        default_store: Option<String>,
        default_model: String,
        system_prompt: Option<String>,
    ) -> Self {
        Self {
            client,
            default_store,
            default_model,
            system_prompt,
            tool_router: Self::tool_router(),
        }
    }
}

#[tool_router(router = tool_router)]
impl GeminiMcpServer {
    #[tool(
        name = "list_file_search_stores",
        description = "List Gemini File Search stores available to the configured API key."
    )]
    async fn list_file_search_stores(&self) -> Result<Json<ListStoresOutput>, String> {
        let stores = self
            .client
            .list_stores()
            .await
            .map_err(|error| tool_error("failed to list File Search stores", error))?
            .into_iter()
            .map(|store| StoreSummary {
                name: store.name,
                display_name: store.display_name,
                embedding_model: store.embedding_model,
            })
            .collect();

        Ok(Json(ListStoresOutput { stores }))
    }

    #[tool(
        name = "list_gemini_models",
        description = "List Gemini models that support content generation."
    )]
    async fn list_gemini_models(&self) -> Result<Json<ListModelsOutput>, String> {
        let models = self
            .client
            .list_models()
            .await
            .map_err(|error| tool_error("failed to list Gemini models", error))?
            .into_iter()
            .filter(|model| model.supports_generate_content())
            .map(|model| ModelSummary {
                name: model.name,
                display_name: model.display_name,
            })
            .collect();

        Ok(Json(ListModelsOutput { models }))
    }

    #[tool(
        name = "query_file_search_store",
        description = "Answer a question with Gemini, grounded in a File Search store. Returns the answer, retrieved file references, and native Gemini token usage."
    )]
    async fn query_file_search_store(
        &self,
        Parameters(input): Parameters<QueryInput>,
    ) -> Result<Json<QueryOutput>, String> {
        let prompt = input.prompt.trim();
        if prompt.is_empty() {
            return Err("prompt must not be empty".to_string());
        }
        let store = non_empty(input.store.as_deref())
            .or_else(|| non_empty(self.default_store.as_deref()))
            .ok_or_else(|| {
                "store is required; pass it to the tool or configure GEMINI_FILE_SEARCH_STORE"
                    .to_string()
            })?;
        let model = normalize_model_name(
            non_empty(input.model.as_deref()).unwrap_or(self.default_model.as_str()),
        );

        logging::event(format!(
            "MCP query: store={store} model={model} prompt_chars={} system_prompt_chars={}",
            prompt.chars().count(),
            self.system_prompt
                .as_deref()
                .map(str::chars)
                .map(Iterator::count)
                .unwrap_or(0)
        ));
        let response = self
            .client
            .generate_content(&model, store, prompt, self.system_prompt.as_deref())
            .await
            .map_err(|error| tool_error("Gemini File Search query failed", error))?;
        query_output(&response, store.to_string(), model).map(Json)
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for GeminiMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("gemini-rag", env!("CARGO_PKG_VERSION")))
            .with_instructions(
                "Read-only Gemini File Search tools for listing stores and models and answering grounded questions.",
            )
    }
}

pub async fn serve_mcp(client: GeminiClient, args: McpArgs) -> Result<()> {
    let bind: SocketAddr = args
        .bind
        .parse()
        .with_context(|| format!("invalid MCP bind address: {}", args.bind))?;
    let token = args.token.trim();
    let allowed_hosts = args
        .allowed_hosts
        .iter()
        .map(|host| host.trim())
        .filter(|host| !host.is_empty())
        .map(str::to_string)
        .collect::<Vec<_>>();
    validate_server_config(bind, token, &allowed_hosts)?;

    let system_prompt = read_optional_system_prompt(args.system_prompt_file.as_deref()).await?;
    let default_model = non_empty(Some(&args.model))
        .map(normalize_model_name)
        .ok_or_else(|| anyhow::anyhow!("MCP default model must not be empty"))?;
    let default_store_label = args.store.as_deref().unwrap_or("<none>").to_string();
    let handler = GeminiMcpServer::new(client, args.store, default_model.clone(), system_prompt);
    let mut config = StreamableHttpServerConfig::default();
    if !allowed_hosts.is_empty() {
        config = config.with_allowed_hosts(allowed_hosts.iter().map(String::as_str));
    }
    let app = mcp_router(handler, token.to_string(), config);
    let listener = tokio::net::TcpListener::bind(bind)
        .await
        .with_context(|| format!("failed to bind MCP server to {bind}"))?;

    logging::event(format!(
        "MCP server listening: bind={bind} path=/mcp default_store={} default_model={} allowed_hosts={} auth_token_present=true",
        default_store_label,
        default_model,
        allowed_hosts.len(),
    ));
    println!("Gemini RAG MCP server listening on http://{bind}/mcp");
    axum::serve(listener, app)
        .await
        .context("MCP server failed")
}

fn validate_server_config(bind: SocketAddr, token: &str, allowed_hosts: &[String]) -> Result<()> {
    if token.is_empty() {
        bail!("GEMINI_MCP_TOKEN or --token must not be empty");
    }
    if !bind.ip().is_loopback() && allowed_hosts.is_empty() {
        bail!(
            "GEMINI_MCP_ALLOWED_HOSTS or --allowed-host is required for a non-loopback MCP bind address"
        );
    }

    Ok(())
}

fn mcp_router(
    handler: GeminiMcpServer,
    token: String,
    config: StreamableHttpServerConfig,
) -> Router {
    let service: StreamableHttpService<GeminiMcpServer, LocalSessionManager> =
        StreamableHttpService::new(move || Ok(handler.clone()), Default::default(), config);

    Router::new()
        .nest_service("/mcp", service)
        .layer(middleware::from_fn_with_state(
            Arc::new(token),
            require_bearer,
        ))
}

async fn require_bearer(
    State(expected): State<Arc<String>>,
    request: Request<Body>,
    next: Next,
) -> Response {
    if is_authorized(request.headers(), expected.as_str()) {
        next.run(request).await
    } else {
        (
            StatusCode::UNAUTHORIZED,
            [(header::WWW_AUTHENTICATE, "Bearer")],
            "Unauthorized",
        )
            .into_response()
    }
}

fn is_authorized(headers: &axum::http::HeaderMap, expected: &str) -> bool {
    headers
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
        .is_some_and(|provided| constant_time_eq(provided.as_bytes(), expected.as_bytes()))
}

fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    if left.len() != right.len() {
        return false;
    }

    left.iter()
        .zip(right)
        .fold(0_u8, |difference, (left, right)| {
            difference | (left ^ right)
        })
        == 0
}

fn normalize_model_name(model: &str) -> String {
    match model.strip_prefix("models/").unwrap_or(model) {
        "gemini-flash-3-preview" => "gemini-3-flash-preview".to_string(),
        model => model.to_string(),
    }
}

fn non_empty(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|value| !value.is_empty())
}

fn tool_error(context: &str, error: anyhow::Error) -> String {
    logging::error(format!("{context}: {error:#}"));
    format!("{context}: {error:#}")
}

fn query_output(
    response: &GenerateContentResponse,
    store: String,
    model: String,
) -> Result<QueryOutput, String> {
    let answer = response
        .text()
        .ok_or_else(|| "Gemini response did not include answer text".to_string())?;
    let references = file_references(std::slice::from_ref(response));
    let usage = response
        .usage_metadata
        .as_ref()
        .and_then(|usage| serde_json::to_value(usage).ok());

    Ok(QueryOutput {
        answer,
        store,
        model,
        references,
        usage,
    })
}

async fn read_optional_system_prompt(path: Option<&Path>) -> Result<Option<String>> {
    let Some(path) = path else {
        return Ok(None);
    };
    if path.as_os_str().is_empty() {
        return Ok(None);
    }

    let prompt = tokio::fs::read_to_string(path)
        .await
        .with_context(|| format!("failed to read MCP system prompt file {}", path.display()))?;
    if prompt.trim().is_empty() {
        bail!("MCP system prompt file is empty: {}", path.display());
    }

    Ok(Some(prompt.trim().to_string()))
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use axum::http::{HeaderMap, HeaderValue, header};
    use rmcp::transport::streamable_http_server::StreamableHttpServerConfig;
    use serde_json::json;

    use crate::{cli::DEFAULT_BASE_URL, gemini::GeminiClient};

    use super::{
        GeminiMcpServer, constant_time_eq, is_authorized, mcp_router, non_empty,
        normalize_model_name, query_output, validate_server_config,
    };

    #[test]
    fn bearer_auth_requires_exact_token() {
        let mut headers = HeaderMap::new();
        assert!(!is_authorized(&headers, "secret"));

        headers.insert(
            header::AUTHORIZATION,
            HeaderValue::from_static("Basic secret"),
        );
        assert!(!is_authorized(&headers, "secret"));

        headers.insert(
            header::AUTHORIZATION,
            HeaderValue::from_static("Bearer wrong"),
        );
        assert!(!is_authorized(&headers, "secret"));

        headers.insert(
            header::AUTHORIZATION,
            HeaderValue::from_static("Bearer secret"),
        );
        assert!(is_authorized(&headers, "secret"));
    }

    #[test]
    fn token_comparison_checks_length_and_contents() {
        assert!(constant_time_eq(b"secret", b"secret"));
        assert!(!constant_time_eq(b"secret", b"secrex"));
        assert!(!constant_time_eq(b"secret", b"short"));
    }

    #[test]
    fn query_defaults_normalize_empty_values_and_model_aliases() {
        assert_eq!(non_empty(Some("  demo  ")), Some("demo"));
        assert_eq!(non_empty(Some("  ")), None);
        assert_eq!(non_empty(None), None);
        assert_eq!(
            normalize_model_name("models/gemini-flash-3-preview"),
            "gemini-3-flash-preview"
        );
    }

    #[test]
    fn server_config_requires_token_and_allowed_hosts_for_remote_bind() {
        let loopback = "127.0.0.1:8090".parse().expect("loopback address");
        let remote = "0.0.0.0:8090".parse().expect("remote address");

        assert!(validate_server_config(loopback, "secret", &[]).is_ok());
        assert!(validate_server_config(loopback, "", &[]).is_err());
        assert!(validate_server_config(remote, "secret", &[]).is_err());
        assert!(
            validate_server_config(remote, "secret", &["mcp.example.test".to_string()]).is_ok()
        );
    }

    #[test]
    fn tool_router_exposes_only_read_and_query_tools_with_output_schemas() {
        let server = test_server();
        let tools = server.tool_router.list_all();
        let names = tools
            .iter()
            .map(|tool| tool.name.as_ref())
            .collect::<Vec<_>>();

        assert_eq!(
            names,
            [
                "list_file_search_stores",
                "list_gemini_models",
                "query_file_search_store"
            ]
        );
        assert!(tools.iter().all(|tool| tool.output_schema.is_some()));
    }

    #[test]
    fn query_output_includes_answer_references_and_native_usage() {
        let response = serde_json::from_value(json!({
            "candidates": [{
                "content": { "parts": [{ "text": "Grounded answer" }] },
                "groundingMetadata": {
                    "groundingChunks": [{
                        "retrievedContext": {
                            "title": "Source",
                            "text": "Relevant excerpt",
                            "fileSearchStore": "fileSearchStores/demo"
                        }
                    }]
                }
            }],
            "usageMetadata": {
                "promptTokenCount": 12,
                "candidatesTokenCount": 4,
                "totalTokenCount": 16
            }
        }))
        .expect("Gemini response");

        let output = query_output(
            &response,
            "fileSearchStores/demo".to_string(),
            "gemini-3-flash-preview".to_string(),
        )
        .expect("query output");
        let value = serde_json::to_value(output).expect("serialized output");

        assert_eq!(value["answer"], "Grounded answer");
        assert_eq!(value["store"], "fileSearchStores/demo");
        assert_eq!(value["model"], "gemini-3-flash-preview");
        assert_eq!(value["references"][0]["title"], "Source");
        assert_eq!(value["usage"]["promptTokenCount"], 12);
        assert_eq!(value["usage"]["totalTokenCount"], 16);
    }

    #[test]
    fn query_output_rejects_response_without_answer_text() {
        let response = serde_json::from_value(json!({})).expect("Gemini response");

        let error = query_output(
            &response,
            "fileSearchStores/demo".to_string(),
            "gemini-3-flash-preview".to_string(),
        )
        .expect_err("missing answer should fail");

        assert_eq!(error, "Gemini response did not include answer text");
    }

    #[tokio::test]
    async fn http_endpoint_requires_bearer_auth_and_accepts_initialize() {
        let app = mcp_router(
            test_server(),
            "secret".to_string(),
            StreamableHttpServerConfig::default().with_sse_keep_alive(None),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener");
        let address = listener.local_addr().expect("listener address");
        let task = tokio::spawn(async move { axum::serve(listener, app).await });
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
            .expect("HTTP client");
        let url = format!("http://{address}/mcp");
        let initialize = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": { "name": "gemini-rag-test", "version": "1.0" }
            }
        });

        let unauthorized = client
            .post(&url)
            .header("Accept", "application/json, text/event-stream")
            .json(&initialize)
            .send()
            .await
            .expect("unauthorized request");
        assert_eq!(unauthorized.status(), reqwest::StatusCode::UNAUTHORIZED);
        assert_eq!(
            unauthorized
                .headers()
                .get(header::WWW_AUTHENTICATE)
                .and_then(|value| value.to_str().ok()),
            Some("Bearer")
        );

        let authorized = client
            .post(&url)
            .bearer_auth("secret")
            .header("Accept", "application/json, text/event-stream")
            .json(&initialize)
            .send()
            .await
            .expect("authorized request");
        assert_eq!(authorized.status(), reqwest::StatusCode::OK);
        let session_id = authorized
            .headers()
            .get("mcp-session-id")
            .and_then(|value| value.to_str().ok())
            .expect("MCP session ID")
            .to_string();
        let initialize_body = authorized.text().await.expect("initialize response body");
        assert!(initialize_body.contains("protocolVersion"));
        assert!(initialize_body.contains("gemini-rag"));

        let initialized = client
            .post(&url)
            .bearer_auth("secret")
            .header("Accept", "application/json, text/event-stream")
            .header("MCP-Protocol-Version", "2025-11-25")
            .header("Mcp-Session-Id", &session_id)
            .json(&json!({
                "jsonrpc": "2.0",
                "method": "notifications/initialized"
            }))
            .send()
            .await
            .expect("initialized notification");
        assert!(initialized.status().is_success());

        let tools = client
            .post(&url)
            .bearer_auth("secret")
            .header("Accept", "application/json, text/event-stream")
            .header("MCP-Protocol-Version", "2025-11-25")
            .header("Mcp-Session-Id", &session_id)
            .json(&json!({
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list",
                "params": {}
            }))
            .send()
            .await
            .expect("tools/list request");
        assert_eq!(tools.status(), reqwest::StatusCode::OK);
        let tools_body = tools.text().await.expect("tools/list response body");
        assert!(tools_body.contains("list_file_search_stores"));
        assert!(tools_body.contains("list_gemini_models"));
        assert!(tools_body.contains("query_file_search_store"));

        task.abort();
    }

    fn test_server() -> GeminiMcpServer {
        let client = GeminiClient::new(Some("test-key".to_string()), DEFAULT_BASE_URL.to_string())
            .expect("Gemini client");
        GeminiMcpServer::new(
            client,
            Some("fileSearchStores/demo".to_string()),
            "gemini-3-flash-preview".to_string(),
            None,
        )
    }
}
