use std::{io, path::PathBuf, process};

use clap::Parser;
use thiserror::Error;
use tokio::net::TcpListener;
use venice_e2ee_proxy::{
    config::{ConfigError, ProxyConfig},
    http,
    venice::VeniceClientError,
};

#[derive(Debug, Parser)]
#[command(
    name = "venice-e2ee-proxy",
    about = "Local OpenAI-compatible proxy shell for Venice E2EE models"
)]
struct Cli {
    /// Path to a TOML configuration file.
    config: PathBuf,
}

#[tokio::main]
async fn main() {
    if let Err(error) = run().await {
        eprintln!("venice-e2ee-proxy: {error}");
        process::exit(1);
    }
}

async fn run() -> Result<(), RunError> {
    let cli = Cli::parse();
    let config = ProxyConfig::load_from_path(cli.config)?;
    let bind_host = config.server.host.clone();
    let bind_port = config.server.port;
    let app = http::router(config)?;
    let listener = TcpListener::bind((bind_host.as_str(), bind_port)).await?;
    let local_addr = listener.local_addr()?;

    eprintln!("venice-e2ee-proxy listening on http://{local_addr}");
    http::serve(listener, app).await?;
    Ok(())
}

#[derive(Debug, Error)]
enum RunError {
    #[error(transparent)]
    Config(#[from] ConfigError),
    #[error(transparent)]
    VeniceClient(#[from] VeniceClientError),
    #[error(transparent)]
    Io(#[from] io::Error),
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
