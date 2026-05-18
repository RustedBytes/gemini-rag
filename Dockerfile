FROM rust:1.95-bookworm AS builder

WORKDIR /app

COPY Cargo.toml Cargo.lock ./
COPY .cargo ./.cargo
COPY src ./src

ARG CPU_BASELINE=x86-64
ENV RUSTFLAGS="-C target-cpu=${CPU_BASELINE}"
ENV CFLAGS="-march=${CPU_BASELINE} -mtune=generic"
ENV CXXFLAGS="-march=${CPU_BASELINE} -mtune=generic"

RUN cargo build --release --locked

FROM debian:bookworm-slim AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /usr/sbin/nologin app \
    && mkdir -p /data \
    && chown app:app /data

WORKDIR /app
COPY --from=builder /app/target/release/gemini-rag /usr/local/bin/gemini-rag

USER app
EXPOSE 8080

ENV GEMINI_RAG_BIND=0.0.0.0:8080
ENV GEMINI_RAG_LOG=/data/gemini-rag.log

CMD ["gemini-rag", "serve"]
