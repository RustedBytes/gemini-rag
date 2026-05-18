# gemini-rag

- Small Rust CLI for feeding local files into a Gemini File Search store and querying them with a selected Gemini model.
- A proxy (as OpenAI compatible API) to Gemini models + included File Store for RAG

## Setup

```bash
cargo build
export GEMINI_API_KEY="your-api-key"
```

Local `.env` files are loaded automatically before CLI arguments are parsed, so
you can also keep development settings there:

```bash
GEMINI_API_KEY="your-api-key"
GEMINI_FILE_SEARCH_STORE="fileSearchStores/ragdocs-abc123"
RUST_LOG=gemini_rag=debug,reqwest=info
```

Builds default to a generic `x86-64` CPU baseline so release binaries run on older
64-bit x86 CPUs instead of requiring newer AVX-era instructions. To target a
newer baseline intentionally, override `RUSTFLAGS`/`CFLAGS`, or pass a Docker
build arg:

```bash
docker build --build-arg CPU_BASELINE=x86-64-v2 .
```

Operations are appended to `gemini-rag.log` by default. Override the path with:

```bash
export GEMINI_RAG_LOG="./rag.log"
# or pass --log-file ./rag.log
```

Runtime logs also go through `env_logger` on stderr. Use `RUST_LOG=debug` for
verbose application logs, or target this crate specifically:

```bash
RUST_LOG=gemini_rag=debug ./target/debug/gemini-rag list-models
RUST_LOG=gemini_rag=trace,reqwest=info ./target/debug/gemini-rag serve
```

For PDF page ingestion, install `pdftoppm`:

```bash
sudo apt install poppler-utils
```

## Create a Store

```bash
./target/debug/gemini-rag create-store --display-name lawdocs
```

The command prints a store name like:

```text
fileSearchStores/lawdocs-abc123
```

You can reuse it with:

```bash
export GEMINI_FILE_SEARCH_STORE="fileSearchStores/lawdocs-abc123"
```

## Ingest a Folder

```bash
./target/debug/gemini-rag ingest ./docs --store "$GEMINI_FILE_SEARCH_STORE"
```

Useful options:

```bash
--no-recursive
--include-hidden
--max-bytes 10000000
--no-wait
```

## Ingest a PDF

Each PDF page is rendered to JPEG, OCR text is extracted with Gemini, and the page text is uploaded into the selected store.

```bash
./target/debug/gemini-rag ingest-pdf ./law.pdf \
  --store "$GEMINI_FILE_SEARCH_STORE" \
  --dpi 200
```

Test a page range before running a large PDF:

```bash
./target/debug/gemini-rag ingest-pdf ./law.pdf \
  --store "$GEMINI_FILE_SEARCH_STORE" \
  --first-page 1 \
  --last-page 1
```

## Query

```bash
./target/debug/gemini-rag query \
  --store "$GEMINI_FILE_SEARCH_STORE" \
  --model gemini-3-flash-preview \
  "Який закон регулює розірвання шлюбів?"
```

Use a system prompt from a file when you want consistent answering rules:

```bash
./target/debug/gemini-rag query \
  --store "$GEMINI_FILE_SEARCH_STORE" \
  --system-prompt-file ./system-prompt.txt \
  "What does this corpus say about divorce?"
```

Show retrieved citation chunks:

```bash
./target/debug/gemini-rag query \
  --store "$GEMINI_FILE_SEARCH_STORE" \
  --show-citations \
  "What does this corpus say about divorce?"
```

## OpenAI-Compatible Proxy

Serve an Axum HTTP API that accepts OpenAI-style chat completions and answers through Gemini. When `GEMINI_FILE_SEARCH_STORE` is set, responses are grounded in that File Search store and retrieved chunks are appended as markdown citations.

```bash
export GEMINI_API_KEY="your-api-key"
export GEMINI_FILE_SEARCH_STORE="fileSearchStores/ragdocs-l5tonm93ebb2"
export GEMINI_PROXY_MODEL="gemini-3-flash-preview"
export GEMINI_SYSTEM_PROMPT_FILE="./system-prompt.txt"

./target/debug/gemini-rag serve --bind 127.0.0.1:8080
```

`GEMINI_PROXY_MODEL` selects the Gemini model used by the server. The OpenAI request's `model` field is accepted for compatibility, but the proxy uses this server setting when calling Gemini.
`GEMINI_SYSTEM_PROMPT_FILE` is optional and is read once at server startup.

Call it with any OpenAI-compatible client by pointing the base URL at the proxy:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gemini-3-flash-preview",
    "messages": [
      { "role": "user", "content": "What does this corpus say about divorce?" }
    ]
  }'
```

Streaming chat completions are supported with OpenAI-compatible server-sent events:

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gemini-3-flash-preview",
    "stream": true,
    "messages": [
      { "role": "user", "content": "What does this corpus say about divorce?" }
    ]
  }'
```

You can override the default store per request:

```json
{
  "model": "gemini-3-flash-preview",
  "store": "fileSearchStores/another-store",
  "messages": [
    { "role": "user", "content": "What does this corpus say about divorce?" }
  ]
}
```

## Docker

Create a local `.env` from the example and fill in your Gemini settings:

```bash
cp .env.example .env
```

Start the OpenAI-compatible server:

```bash
docker compose up --build
# or, with the legacy Compose binary:
docker-compose up --build
```

If legacy `docker-compose` fails with `KeyError: 'ContainerConfig'` while recreating the service, remove the stale container first:

```bash
docker-compose down --remove-orphans
docker rm -f gemini-proxy-with-filestore_openai-server_1 2>/dev/null || true
docker-compose up --build --force-recreate
```

The server listens on:

```text
http://127.0.0.1:8080/v1
```

Test it:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "ignored-by-proxy",
    "messages": [
      { "role": "user", "content": "What does this corpus say about divorce?" }
    ]
  }'
```

Set `OPENAI_SERVER_PORT` in `.env` if you want to publish a different host port. Logs are written to `/data/gemini-rag.log` inside the `gemini-rag-data` Docker volume.
If you set `GEMINI_SYSTEM_PROMPT_FILE` for Docker, make sure the file path exists
inside the container, for example by bind-mounting it to `/data/system-prompt.txt`.

## Releases

Pushing a Git tag triggers the release workflow. It builds the release binary for
a generic x86-64 Linux CPU baseline on Ubuntu 22.04, keeping the glibc
requirement at `GLIBC_2.35` or older for environments such as Google Colab. The
workflow uploads a `.tar.gz` plus SHA-256 checksum to the matching GitHub
Release.

```bash
git tag v0.1.0
git push origin v0.1.0
```

## Other Commands

```bash
./target/debug/gemini-rag list-stores
./target/debug/gemini-rag list-models
./target/debug/gemini-rag delete-store --store "$GEMINI_FILE_SEARCH_STORE" --force
```

## License

Apache-2.0. Copyright 2026 Yehor Smoliakov <egorsmkv@gmail.com>.
