pub mod config;
pub mod data;
pub mod error;
pub mod federated;
pub mod ingest;
pub mod metrics;
pub mod node;
pub mod preprocess;
pub mod sync;
pub mod tensor;

pub use config::{Config, NodeConfig};
pub use data::{AstroData, DataBatch, DataSource};
pub use error::{AstroError, Result};
pub use federated::{FederatedNode, NodeStatus};
pub use ingest::{DataIngester, IngestionStats};
pub use metrics::{MetricsCollector, NodeMetrics};
pub use node::Node;
pub use preprocess::{PreprocessingPipeline, Preprocessor};
pub use tensor::{TensorBatch, TensorEncoder};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub fn init_logging() {
    use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

    tracing_subscriber::registry()
        .with(EnvFilter::from_default_env())
        .with(tracing_subscriber::fmt::layer())
        .init();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}
