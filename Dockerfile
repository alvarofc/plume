FROM rust:1.92-bookworm AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y protobuf-compiler && rm -rf /var/lib/apt/lists/*

COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/
COPY bench/ bench/

RUN cargo build --release --bin plume

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/plume /usr/local/bin/plume
COPY config.toml /app/config.toml

ENV PLUME_CONFIG=/app/config.toml

EXPOSE 3000

CMD ["plume"]
