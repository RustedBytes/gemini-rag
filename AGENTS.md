# AGENTS.md - Development Guide for gemini-rag

This guide is for AI coding agents working in this repository. Keep it accurate as the project changes; it should describe how to make safe edits, run the right checks, and respect the project boundaries.

## Project Overview

`gemini-rag` is a Rust 2024 CLI and HTTP service for Gemini File Search workflows:

- Create, list, query, and delete Gemini File Search stores.
- Ingest local folders into a store, including image-aware store selection.
- Render PDF pages with `pdftoppm`, optionally OCR them through Gemini, and upload page documents.
- Serve an OpenAI-compatible chat completions proxy backed by Gemini and optional File Search grounding.

The crate is intentionally small and mostly organized by user-facing capability. Prefer direct, idiomatic Rust over new framework layers.

## Technology Stack

- Rust 2024 edition.
- Async runtime: `tokio`.
- CLI: `clap` with environment variable support.
- HTTP server: `axum`.
- HTTP client: `reqwest` using `rustls-tls`.
- Serialization: `serde` and `serde_json`.
- Errors: `anyhow` in application code; custom API errors in `src/server/error.rs`.
- Logging: local file logging through `src/logging.rs`, plus `env_logger` on stderr.
- PDF rendering: external `pdftoppm` from `poppler-utils`.
- Docker: multi-stage build from `rust:1.95-bookworm` to `debian:bookworm-slim`.

Release binaries intentionally target a generic `x86-64` CPU baseline. Do not remove or casually override `.cargo/config.toml`, Docker `CPU_BASELINE`, release workflow `RUSTFLAGS`, or the glibc compatibility check.

## Repository Layout

```text
.
|-- Cargo.toml              # Crate metadata, dependencies, release profile
|-- Cargo.lock              # Locked dependency graph; keep committed
|-- README.md               # User setup and command documentation
|-- system-prompt.txt       # Example/default prompt packaged in releases
|-- Dockerfile              # Release container image
|-- docker-compose.yml      # Local OpenAI-compatible server deployment
|-- .cargo/config.toml      # Generic x86-64 build defaults
|-- .github/workflows/
|   `-- release.yml         # Tag-driven release build and upload
`-- src/
    |-- main.rs             # Entry point, dotenv load, command dispatch
    |-- cli.rs              # Clap commands, flags, env vars, defaults
    |-- gemini/             # Gemini REST client, API types, upload helpers
    |-- server/             # Axum OpenAI-compatible API, SSE, citations, types
    |-- ingest.rs           # Folder ingestion workflow
    |-- pdf.rs              # PDF render/OCR/upload workflow
    |-- query.rs            # Grounded CLI query workflow
    |-- files.rs            # Local file collection rules
    |-- output.rs           # CLI output formatting
    `-- logging.rs          # File and stderr logging setup/helpers
```

## Common Commands

Use these from the repository root.

```bash
cargo fmt --all -- --check
cargo check --locked
cargo clippy --locked --all-targets -- -D warnings
cargo build --locked
cargo build --release --locked
```

There is currently no dedicated test suite in the repository. When changing pure logic, add focused unit tests near the code if practical. When changing Gemini API behavior or the proxy, at minimum run `cargo check --locked` and manually exercise the relevant command when credentials are available.

Useful runtime commands:

```bash
cargo run --locked -- list-models
cargo run --locked -- create-store --display-name rag-docs
cargo run --locked -- ingest ./docs --store "$GEMINI_FILE_SEARCH_STORE"
cargo run --locked -- ingest-pdf ./law.pdf --store "$GEMINI_FILE_SEARCH_STORE" --first-page 1 --last-page 1
cargo run --locked -- query --store "$GEMINI_FILE_SEARCH_STORE" "What does this corpus say?"
cargo run --locked -- serve --bind 127.0.0.1:8080
```

Docker server workflow:

```bash
cp .env.example .env
docker compose up --build
```

## Environment Variables

The CLI loads `.env` before parsing arguments. Never commit real API keys or generated logs.

- `GEMINI_API_KEY`: required for all Gemini operations.
- `GEMINI_BASE_URL`: defaults to `https://generativelanguage.googleapis.com`.
- `GEMINI_FILE_SEARCH_STORE`: default store for ingest/query/server commands.
- `GEMINI_PROXY_MODEL`: server-side model used by the proxy.
- `GEMINI_SYSTEM_PROMPT_FILE`: optional prompt file read once at command/server startup.
- `GEMINI_RAG_BIND`: server bind address.
- `GEMINI_RAG_LOG`: log file path; defaults to `gemini-rag.log`.
- `RUST_LOG`: stderr logging filter, for example `gemini_rag=debug,reqwest=info`.

## Development Guidelines

- Keep CLI surface changes centralized in `src/cli.rs`, then update `README.md` and this file when flags, env vars, defaults, or commands change.
- Keep Gemini API request/response handling in `src/gemini/`. Avoid leaking raw request construction into CLI or server modules.
- Keep OpenAI-compatible schema and request parsing in `src/server/types.rs`; keep Axum routing and handler behavior in `src/server/mod.rs`.
- Preserve support for both `/v1/chat/completions` and `/chat/completions` unless deliberately making a compatibility-breaking change.
- Preserve streaming behavior in `src/server/sse.rs` when changing non-streaming chat completions.
- Use `anyhow::Context` for fallible external operations so user errors include the path, store, model, or command involved.
- Prefer `bail!` for validation failures and keep messages suitable for CLI users.
- Do not log secret values. The existing client logs API key presence and length only; keep it that way.
- Keep operation logs useful but not noisy: event-level messages for high-level workflow progress, debug-level messages for request details.
- Avoid changing `Cargo.lock` unless dependency changes require it.

## Gemini and File Search Notes

- Store names usually look like `fileSearchStores/name-suffix`; accept and pass through full store names.
- `DEFAULT_MODEL` is currently `gemini-3-flash-preview`.
- The query path normalizes `gemini-flash-3-preview` to `gemini-3-flash-preview`; server model normalization lives separately in `src/server/util.rs`.
- Folder ingestion creates a store when `--store` is omitted. If JPEG/PNG files are present, it uses `models/gemini-embedding-2` so image File Search works.
- Existing stores used for image ingestion must already use `models/gemini-embedding-2`.
- Folder and PDF ingestion sleep between upload batches by default (`--upload-delay-secs 1`) to reduce API pressure; use `0` to disable when appropriate.
- PDF ingestion depends on `pdftoppm`. Do not replace that with a Rust PDF renderer without a clear reason and README updates.
- PDF OCR writes temporary text documents with source image and page metadata before uploading.

## OpenAI-Compatible Proxy Notes

The server exposes:

- `GET /healthz`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /chat/completions`

OpenAI request `model` is accepted. The server currently normalizes and uses the request model when present; if omitted it falls back to `GEMINI_PROXY_MODEL` or the CLI default. Requests may override the default store through `store`, `file_search_store`, or `fileSearchStore`.

Responses include Gemini metadata and file references under `metadata`. Non-streaming image responses can also include message-level `metadata.images`.

## Formatting, Style, and Safety

- Run `cargo fmt` before finishing Rust edits.
- Keep Rust simple and idiomatic. Prefer small functions, explicit data flow, and straightforward `match`/`if let` handling over clever combinator chains when the latter hurts readability.
- Prefer borrowing (`&str`, `&Path`, slices) at API boundaries when ownership is not needed. Allocate `String`, `PathBuf`, and `Vec` only when a value must be owned or accumulated.
- Use `Option` and `Result` directly. Avoid sentinel values, boolean-plus-output patterns, and panics for expected user, filesystem, network, or API failures.
- Use `?` with `anyhow::Context` to preserve the original error and add local meaning. Keep validation errors direct with `bail!`.
- Keep async code idiomatic: do not block inside async request paths except for deliberate external process work such as `pdftoppm`; use `tokio::fs` and async reqwest calls where practical.
- Prefer existing helpers and module boundaries before adding abstractions. Add a new abstraction only when it removes real duplication or clarifies a shared contract.
- Keep structs and serde types close to the boundary they model. Publicly expose only the types needed by other modules.
- Avoid unnecessary clones. Clone intentionally at async/task/state boundaries, and prefer `Arc` only when shared ownership is actually required.
- Preserve strong typing for API shapes instead of building JSON with ad hoc string manipulation. Use `serde_json::json!` for small request bodies and typed structs for reusable schemas.
- Keep comments sparse and useful. The codebase mostly relies on clear function names, typed boundaries, and contextual errors.
- Use async file and network APIs inside async workflows.
- Avoid panics in command/server paths. Existing `unwrap_or` fallbacks are fine for non-fallible formatting, but new user-input or network paths should return errors.
- Keep output stable for CLI users; when changing printed text, update README examples if they become inaccurate.
- Keep generated artifacts, local `.env`, `gemini-rag.log`, and `target/` out of commits.

## Verification Checklist

Before handing off a change, run the narrowest meaningful set:

```bash
cargo fmt --all -- --check
cargo check --locked
```

Also run these when relevant:

```bash
cargo clippy --locked --all-targets -- -D warnings
cargo build --release --locked
docker compose config
```

For API-affecting work, manually test with real credentials when available and mention if you could not. For PDF work, test at least a one-page range with `--first-page 1 --last-page 1` when a sample PDF and `pdftoppm` are available.

## Documentation Expectations

Update `README.md` for user-facing changes:

- New or renamed commands, flags, environment variables, endpoints, or response fields.
- Changed defaults for model, bind address, log path, or CPU baseline.
- New Docker volume, port, build argument, or runtime requirement.
- New external tools required at runtime.

Update this `AGENTS.md` when internal workflows, module ownership, or verification commands change.
