use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::process;

use astroscale_core::{init_logging, Config, Result, VERSION};

#[derive(Parser)]
#[command(name = "astroscale-node")]
#[command(author = "AstroScale Team")]
#[command(version = VERSION)]
#[command(about = "High-performance distributed astronomical data processing", long_about = None)]
struct Cli {
    #[arg(short, long, default_value = "config.toml")]
    config: PathBuf,

    #[arg(long)]
    node_id: Option<String>,

    #[arg(short, long)]
    log_level: Option<String>,

    #[arg(short, long)]
    verbose: bool,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    Start {
        #[arg(short, long)]
        daemon: bool,
    },

    Config {
        #[arg(short, long)]
        show: bool,
    },

    Ingest {
        #[arg(short, long)]
        source: String,

        #[arg(short, long)]
        limit: Option<usize>,
    },

    Preprocess {
        #[arg(short, long)]
        input: PathBuf,

        #[arg(short, long)]
        output: PathBuf,
    },

    Status,

    InitConfig {
        #[arg(short, long, default_value = "config.toml")]
        output: PathBuf,
    },
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    if let Some(level) = &cli.log_level {
        std::env::set_var("RUST_LOG", level);
    } else if cli.verbose {
        std::env::set_var("RUST_LOG", "debug");
    } else {
        std::env::set_var("RUST_LOG", "info");
    }

    init_logging();

    let result = match &cli.command {
        Some(Commands::Start { daemon }) => start_node(&cli, *daemon).await,
        Some(Commands::Config { show }) => validate_config(&cli, *show),
        Some(Commands::Ingest { source, limit }) => ingest_data(&cli, source, *limit).await,
        Some(Commands::Preprocess { input, output }) => preprocess_data(&cli, input, output).await,
        Some(Commands::Status) => show_status(&cli).await,
        Some(Commands::InitConfig { output }) => init_config(output),
        None => start_node(&cli, false).await,
    };

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        process::exit(1);
    }
}

async fn start_node(cli: &Cli, daemon: bool) -> Result<()> {
    println!("Starting AstroScale Node Engine v{}", VERSION);
    println!("");

    let mut config = load_config(cli)?;

    if let Some(node_id) = &cli.node_id {
        config.node.id = node_id.clone();
    }

    config.validate()?;

    println!("Configuration loaded:");
    println!("   Node ID:   {}", config.node.id);
    println!("   Node Name: {}", config.node.name);
    println!("   Role:      {}", config.node.role);
    println!(
        "   Address:   {}:{}",
        config.node.bind_address, config.node.port
    );

    if daemon {
        println!("\nRunning in daemon mode...");
        println!("   (Daemon mode not yet implemented - running in foreground)");
    }

    println!("\nNode started successfully");
    println!(
        "   API: http://{}:{}",
        config.node.bind_address, config.node.port
    );

    if config.node.enable_metrics {
        println!(
            "   Metrics: http://{}:{}",
            config.node.bind_address, config.node.metrics_port
        );
    }

    if config.federated.enabled {
        println!("\nFederated Learning: ENABLED");
        println!(
            "   Mode: {}",
            if config.is_coordinator() {
                "Coordinator"
            } else {
                "Worker"
            }
        );
        if let Some(addr) = &config.federated.coordinator_address {
            println!("   Coordinator: {}", addr);
        }
    }

    println!("\nSystem Information:");
    println!("   Data Directory:  {}", config.storage.data_dir);
    println!("   Cache Directory: {}", config.storage.cache_dir);
    println!("   Model Directory: {}", config.storage.model_dir);

    println!("\nNode is running... Press Ctrl+C to stop");
    println!("\n");

    tokio::signal::ctrl_c()
        .await
        .expect("Failed to listen for ctrl-c");

    println!("\nShutting down gracefully...");
    Ok(())
}

fn validate_config(cli: &Cli, show: bool) -> Result<()> {
    println!("Validating configuration file: {:?}", cli.config);

    let config = load_config(cli)?;
    config.validate()?;

    println!("Configuration is valid");

    if show {
        println!("\nFull Configuration:");
        println!("");
        println!("{:#?}", config);
    } else {
        println!("\nUse --show to display full configuration");
    }

    Ok(())
}

async fn ingest_data(cli: &Cli, source: &str, limit: Option<usize>) -> Result<()> {
    let config = load_config(cli)?;

    println!("Starting data ingestion");
    println!("   Source: {}", source);
    if let Some(limit) = limit {
        println!("   Limit:  {} records", limit);
    }
    println!();

    let source_config = config.ingestion.sources.iter().find(|s| s.name == source);

    match source_config {
        Some(cfg) if cfg.enabled => {
            println!("Source '{}' found in configuration", source);
            println!("   Endpoint: {}", cfg.endpoint);

            println!("\nIngestion system not yet implemented");
            println!("   This is a placeholder for future functionality");
        }
        Some(_) => {
            println!("Source '{}' is disabled in configuration", source);
            return Err(astroscale_core::AstroError::Config(format!(
                "Source '{}' is disabled",
                source
            )));
        }
        None => {
            println!("Source '{}' not found in configuration", source);
            println!("\nAvailable sources:");
            for src in &config.ingestion.sources {
                println!(
                    "   - {} ({})",
                    src.name,
                    if src.enabled { "enabled" } else { "disabled" }
                );
            }
            return Err(astroscale_core::AstroError::NotFound(format!(
                "Source '{}'",
                source
            )));
        }
    }

    Ok(())
}

async fn preprocess_data(cli: &Cli, input: &PathBuf, output: &PathBuf) -> Result<()> {
    let config = load_config(cli)?;

    println!("Starting data preprocessing");
    println!("   Input:  {:?}", input);
    println!("   Output: {:?}", output);
    println!("   Batch Size: {}", config.preprocessing.batch_size);
    println!();

    if !input.exists() {
        return Err(astroscale_core::AstroError::NotFound(format!(
            "Input file: {:?}",
            input
        )));
    }

    if !output.exists() {
        std::fs::create_dir_all(output)?;
        println!("Created output directory: {:?}", output);
    }

    println!("\nPreprocessing pipeline not yet implemented");
    println!("   This is a placeholder for future functionality");

    Ok(())
}

async fn show_status(cli: &Cli) -> Result<()> {
    let config = load_config(cli)?;

    println!("AstroScale Node Status");
    println!("");
    println!("Node ID:     {}", config.node.id);
    println!("Node Name:   {}", config.node.name);
    println!("Role:        {}", config.node.role);
    println!("Status:      Not Running");
    println!("\nRun 'astroscale-node start' to start the node");

    Ok(())
}

fn init_config(output: &PathBuf) -> Result<()> {
    println!("Generating default configuration file");

    if output.exists() {
        println!("File already exists: {:?}", output);
        print!("   Overwrite? [y/N]: ");
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        let mut input = String::new();
        std::io::stdin().read_line(&mut input).unwrap();

        if !input.trim().eq_ignore_ascii_case("y") {
            println!("   Cancelled");
            return Ok(());
        }
    }

    let config = Config::default();

    let toml_string = toml::to_string_pretty(&config)
        .map_err(|e| astroscale_core::AstroError::Serialization(e.to_string()))?;

    std::fs::write(output, toml_string)?;

    println!("Configuration file created: {:?}", output);
    println!("\nNext steps:");
    println!("   1. Edit the configuration file to customize settings");
    println!("   2. Run 'astroscale-node config --show' to validate");
    println!("   3. Run 'astroscale-node start' to start the node");

    Ok(())
}

fn load_config(cli: &Cli) -> Result<Config> {
    if !cli.config.exists() {
        eprintln!("Configuration file not found: {:?}", cli.config);
        eprintln!("\nCreate a default configuration with:");
        eprintln!("   astroscale-node init-config");
        return Err(astroscale_core::AstroError::NotFound(format!(
            "Config file: {:?}",
            cli.config
        )));
    }

    Config::from_file(&cli.config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_parsing() {
        let cli = Cli::parse_from(&["astroscale-node", "--config", "test.toml"]);
        assert_eq!(cli.config, PathBuf::from("test.toml"));
    }
}
