use std::{io, path::PathBuf, process};

use clap::Parser;
use thiserror::Error;
use tokio::net::TcpListener;
use tracing::{error, info};
use tracing_subscriber::{
    EnvFilter, Registry, layer::SubscriberExt, reload, util::SubscriberInitExt,
};
use venice_e2ee_proxy::{
    config::{ConfigError, ProxyConfig},
    http,
    venice::VeniceClientError,
};

/// Command-line arguments for starting the proxy process.
#[derive(Debug, Parser)]
#[command(
    name = "venice-e2ee-proxy",
    about = "Local OpenAI-compatible proxy shell for Venice E2EE models"
)]
struct Cli {
    /// Path to a TOML configuration file.
    config: PathBuf,
}

type EnvFilterReloadHandle = reload::Handle<EnvFilter, Registry>;

/// Starts the async runtime, initializes tracing, and exits with a non-zero status on startup errors.
#[tokio::main]
async fn main() {
    let tracing_reload_handle = init_tracing();

    if let Err(error) = run(&tracing_reload_handle).await {
        error!(%error, "venice-e2ee-proxy exited with error");
        process::exit(1);
    }
}

/// Initializes stdout tracing and returns a handle that can replace the filter after config loads.
fn init_tracing() -> EnvFilterReloadHandle {
    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let (env_filter, reload_handle) = reload::Layer::new(env_filter);

    tracing_subscriber::registry()
        .with(env_filter)
        .with(tracing_subscriber::fmt::layer().with_writer(io::stdout))
        .init();

    reload_handle
}

/// Applies the configured tracing filter string to an existing tracing reload handle.
fn configure_tracing(reload_handle: &EnvFilterReloadHandle, level: &str) -> Result<(), RunError> {
    let env_filter = EnvFilter::try_new(level).map_err(|source| RunError::TracingFilter {
        message: source.to_string(),
    })?;
    reload_handle
        .reload(env_filter)
        .map_err(|source| RunError::TracingReload {
            message: source.to_string(),
        })?;
    Ok(())
}

/// Loads configuration, builds the HTTP router, binds the configured listener, and serves requests.
async fn run(tracing_reload_handle: &EnvFilterReloadHandle) -> Result<(), RunError> {
    let cli = Cli::parse();
    let config = ProxyConfig::load_from_path(&cli.config)?;
    configure_tracing(tracing_reload_handle, &config.logging.level)?;
    info!(config_path = %cli.config.display(), logging_level = %config.logging.level, "configuration loaded");
    let bind_host = config.server.host.clone();
    let bind_port = config.server.port;
    let app = http::router(config)?;
    let listener = TcpListener::bind((bind_host.as_str(), bind_port)).await?;
    let local_addr = listener.local_addr()?;

    info!(address = %local_addr, "venice-e2ee-proxy listening");
    http::serve(listener, app).await?;
    Ok(())
}

/// Startup errors returned while loading configuration, building the router, or serving HTTP traffic.
#[derive(Debug, Error)]
enum RunError {
    #[error(transparent)]
    Config(#[from] ConfigError),
    #[error(transparent)]
    VeniceClient(#[from] VeniceClientError),
    #[error(transparent)]
    Io(#[from] io::Error),
    #[error("invalid tracing filter in logging.level: {message}")]
    TracingFilter { message: String },
    #[error("failed to apply tracing configuration: {message}")]
    TracingReload { message: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_rejects_missing_config_path() {
        let result = Cli::try_parse_from(["venice-e2ee-proxy"]);
        assert!(result.is_err());
    }

    #[test]
    fn cli_accepts_one_config_path() {
        let cli =
            Cli::try_parse_from(["venice-e2ee-proxy", "config.toml"]).expect("CLI should parse");
        assert_eq!(cli.config, PathBuf::from("config.toml"));
    }

    #[test]
    fn cli_rejects_extra_args() {
        let result = Cli::try_parse_from(["venice-e2ee-proxy", "config.toml", "extra"]);
        assert!(result.is_err());
    }
}
