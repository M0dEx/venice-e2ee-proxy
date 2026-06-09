FROM rust:1-alpine AS builder

WORKDIR /app
RUN apk add --no-cache build-base

COPY Cargo.toml Cargo.lock ./
COPY src ./src

RUN cargo build --release --locked

FROM alpine:3.22

RUN apk add --no-cache ca-certificates \
    && addgroup -S proxy \
    && adduser -S -D -H -G proxy proxy \
    && mkdir -p /etc/venice-e2ee-proxy

COPY --chown=proxy:proxy config/default.toml /etc/venice-e2ee-proxy/config.toml
COPY --from=builder /app/target/release/venice-e2ee-proxy /usr/local/bin/venice-e2ee-proxy

USER proxy
EXPOSE 8080

ENTRYPOINT ["venice-e2ee-proxy"]
CMD ["/etc/venice-e2ee-proxy/config.toml"]
