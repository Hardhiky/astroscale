
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use std::time::Duration;

use crate::error::{AstroError, Result};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub node: NodeConfig,

    pub ingestion: IngestionConfig,

    pub preprocessing: PreprocessingConfig,

    #[serde(default)]
    pub federated: FederatedConfig,

    pub storage: StorageConfig,

    pub network: NetworkConfig,

    #[serde(default)]
    pub resources: ResourceConfig,

    #[serde(default)]
    pub logging: LoggingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeConfig {
    pub id: String,

    pub name: String,

    #[serde(default = "default_node_role")]
    pub role: String,

    #[serde(default)]
    pub region: Option<String>,

    #[serde(default = "default_bind_address")]
    pub bind_address: String,

    #[serde(default = "default_port")]
    pub port: u16,

    #[serde(default = "default_true")]
    pub enable_metrics: bool,

    #[serde(default = "default_metrics_port")]
    pub metrics_port: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestionConfig {
    pub sources: Vec<DataSourceConfig>,

    #[serde(default = "default_concurrent_downloads")]
    pub concurrent_downloads: usize,

    #[serde(default = "default_download_timeout")]
    pub download_timeout_secs: u64,

    #[serde(default = "default_retry_attempts")]
    pub retry_attempts: u32,

    #[serde(default = "default_true")]
    pub verify_checksums: bool,

    #[serde(default = "default_true")]
    pub enable_cache: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSourceConfig {
    pub name: String,

    pub endpoint: String,

    #[serde(default)]
    pub auth_token: Option<String>,

    #[serde(default = "default_true")]
    pub enabled: bool,

    #[serde(default)]
    pub rate_limit: Option<f64>,

    #[serde(default)]
    pub headers: std::collections::HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessingConfig {
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,

    #[serde(default = "default_true")]
    pub enable_parallel: bool,

    #[serde(default)]
    pub num_threads: usize,

    #[serde(default = "default_normalization")]
    pub normalization: String,

    #[serde(default = "default_missing_strategy")]
    pub missing_values: String,

    #[serde(default)]
    pub outlier_detection: OutlierConfig,

    #[serde(default)]
    pub feature_engineering: FeatureConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutlierConfig {
    #[serde(default)]
    pub enabled: bool,

    #[serde(default = "default_outlier_method")]
    pub method: String,

    #[serde(default = "default_outlier_threshold")]
    pub threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,

    #[serde(default)]
    pub polynomial_degree: Option<u32>,

    #[serde(default)]
    pub interactions: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedConfig {
    #[serde(default)]
    pub enabled: bool,

    #[serde(default)]
    pub coordinator_address: Option<String>,

    #[serde(default = "default_sync_interval")]
    pub sync_interval_secs: u64,

    #[serde(default = "default_aggregation")]
    pub aggregation_strategy: String,

    #[serde(default = "default_min_nodes")]
    pub min_nodes: usize,

    #[serde(default = "default_local_epochs")]
    pub local_epochs: u32,

    #[serde(default)]
    pub differential_privacy: bool,

    #[serde(default = "default_epsilon")]
    pub epsilon: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    pub data_dir: String,

    pub cache_dir: String,

    pub model_dir: String,

    #[serde(default = "default_cache_size")]
    pub max_cache_size_gb: f64,

    #[serde(default = "default_true")]
    pub enable_compression: bool,

    #[serde(default = "default_compression")]
    pub compression_algorithm: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    #[serde(default = "default_network_timeout")]
    pub timeout_secs: u64,

    #[serde(default = "default_max_connections")]
    pub max_connections: usize,

    #[serde(default = "default_true")]
    pub enable_tls: bool,

    #[serde(default)]
    pub tls_cert_path: Option<String>,

    #[serde(default)]
    pub tls_key_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConfig {
    #[serde(default)]
    pub max_memory_gb: f64,

    #[serde(default)]
    pub max_cpu_cores: usize,

    #[serde(default)]
    pub enable_gpu: bool,

    #[serde(default)]
    pub gpu_devices: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    #[serde(default = "default_log_level")]
    pub level: String,

    #[serde(default = "default_log_format")]
    pub format: String,

    #[serde(default)]
    pub log_to_file: bool,

    #[serde(default)]
    pub log_file_path: Option<String>,
}

fn default_node_role() -> String {
    "worker".to_string()
}

fn default_bind_address() -> String {
    "127.0.0.1".to_string()
}

fn default_port() -> u16 {
    8081
}

fn default_metrics_port() -> u16 {
    9090
}

fn default_concurrent_downloads() -> usize {
    4
}

fn default_download_timeout() -> u64 {
    300 // 5 minutes
}

fn default_retry_attempts() -> u32 {
    3
}

fn default_batch_size() -> usize {
    1024
}

fn default_normalization() -> String {
    "standard".to_string()
}

fn default_missing_strategy() -> String {
    "mean".to_string()
}

fn default_outlier_method() -> String {
    "zscore".to_string()
}

fn default_outlier_threshold() -> f64 {
    3.0
}

fn default_sync_interval() -> u64 {
    300 // 5 minutes
}

fn default_aggregation() -> String {
    "fedavg".to_string()
}

fn default_min_nodes() -> usize {
    2
}

fn default_local_epochs() -> u32 {
    5
}

fn default_epsilon() -> f64 {
    1.0
}

fn default_cache_size() -> f64 {
    10.0 // 10 GB
}

fn default_compression() -> String {
    "zstd".to_string()
}

fn default_network_timeout() -> u64 {
    30
}

fn default_max_connections() -> usize {
    100
}

fn default_log_level() -> String {
    "info".to_string()
}

fn default_log_format() -> String {
    "text".to_string()
}

fn default_true() -> bool {
    true
}

impl Config {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = fs::read_to_string(path.as_ref())
            .map_err(|e| AstroError::Config(format!("Failed to read config file: {}", e)))?;

        Self::from_str(&content)
    }

    pub fn from_str(content: &str) -> Result<Self> {
        toml::from_str(content)
            .map_err(|e| AstroError::Config(format!("Failed to parse config: {}", e)))
    }

    pub fn default() -> Self {
        Self {
            node: NodeConfig {
                id: uuid::Uuid::new_v4().to_string(),
                name: "astroscale-node".to_string(),
                role: "worker".to_string(),
                region: None,
                bind_address: "127.0.0.1".to_string(),
                port: 8081,
                enable_metrics: true,
                metrics_port: 9090,
            },
            ingestion: IngestionConfig {
                sources: vec![],
                concurrent_downloads: 4,
                download_timeout_secs: 300,
                retry_attempts: 3,
                verify_checksums: true,
                enable_cache: true,
            },
            preprocessing: PreprocessingConfig {
                batch_size: 1024,
                enable_parallel: true,
                num_threads: 0,
                normalization: "standard".to_string(),
                missing_values: "mean".to_string(),
                outlier_detection: OutlierConfig::default(),
                feature_engineering: FeatureConfig::default(),
            },
            federated: FederatedConfig::default(),
            storage: StorageConfig {
                data_dir: "./data".to_string(),
                cache_dir: "./cache".to_string(),
                model_dir: "./models".to_string(),
                max_cache_size_gb: 10.0,
                enable_compression: true,
                compression_algorithm: "zstd".to_string(),
            },
            network: NetworkConfig {
                timeout_secs: 30,
                max_connections: 100,
                enable_tls: false,
                tls_cert_path: None,
                tls_key_path: None,
            },
            resources: ResourceConfig::default(),
            logging: LoggingConfig::default(),
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.node.id.is_empty() {
            return Err(AstroError::Config("Node ID cannot be empty".to_string()));
        }

        if self.node.port == 0 {
            return Err(AstroError::Config("Invalid port number".to_string()));
        }

        if self.storage.data_dir.is_empty() {
            return Err(AstroError::Config(
                "Data directory cannot be empty".to_string(),
            ));
        }

        if self.resources.max_memory_gb < 0.0 {
            return Err(AstroError::Config("Invalid memory limit".to_string()));
        }

        if self.federated.enabled {
            if self.node.role == "worker" && self.federated.coordinator_address.is_none() {
                return Err(AstroError::Config(
                    "Worker nodes must specify coordinator address".to_string(),
                ));
            }

            if self.federated.min_nodes < 1 {
                return Err(AstroError::Config(
                    "Minimum nodes must be at least 1".to_string(),
                ));
            }
        }

        Ok(())
    }

    pub fn network_timeout(&self) -> Duration {
        Duration::from_secs(self.network.timeout_secs)
    }

    pub fn sync_interval(&self) -> Duration {
        Duration::from_secs(self.federated.sync_interval_secs)
    }

    pub fn is_coordinator(&self) -> bool {
        self.node.role == "coordinator" || self.node.role == "hybrid"
    }

    pub fn is_worker(&self) -> bool {
        self.node.role == "worker" || self.node.role == "hybrid"
    }
}

impl Default for OutlierConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            method: "zscore".to_string(),
            threshold: 3.0,
        }
    }
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            polynomial_degree: None,
            interactions: false,
        }
    }
}

impl Default for FederatedConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            coordinator_address: None,
            sync_interval_secs: 300,
            aggregation_strategy: "fedavg".to_string(),
            min_nodes: 2,
            local_epochs: 5,
            differential_privacy: false,
            epsilon: 1.0,
        }
    }
}

impl Default for ResourceConfig {
    fn default() -> Self {
        Self {
            max_memory_gb: 0.0,
            max_cpu_cores: 0,
            enable_gpu: false,
            gpu_devices: vec![],
        }
    }
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            format: "text".to_string(),
            log_to_file: false,
            log_file_path: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.node.role, "worker");
        assert_eq!(config.node.port, 8081);
        assert_eq!(config.ingestion.concurrent_downloads, 4);
    }

    #[test]
    fn test_config_validation() {
        let config = Config::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_invalid_config() {
        let mut config = Config::default();
        config.node.id = String::new();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_coordinator_detection() {
        let mut config = Config::default();
        config.node.role = "coordinator".to_string();
        assert!(config.is_coordinator());
        assert!(!config.is_worker());
    }

    #[test]
    fn test_worker_detection() {
        let config = Config::default();
        assert!(!config.is_coordinator());
        assert!(config.is_worker());
    }

    #[test]
    fn test_from_toml() {
        let toml = r#"
            [node]
            id = "test-node"
            name = "Test Node"
            role = "worker"
            bind_address = "0.0.0.0"
            port = 8081

            [ingestion]
            sources = []
            concurrent_downloads = 4

            [preprocessing]
            batch_size = 512

            [storage]
            data_dir = "./data"
            cache_dir = "./cache"
            model_dir = "./models"

            [network]
            timeout_secs = 60
        "#;

        let config = Config::from_str(toml);
        assert!(config.is_ok());
        let config = config.unwrap();
        assert_eq!(config.node.id, "test-node");
        assert_eq!(config.preprocessing.batch_size, 512);
    }
}
