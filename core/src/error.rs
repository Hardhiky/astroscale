
use std::fmt;
use std::io;

pub type Result<T> = std::result::Result<T, AstroError>;

#[derive(Debug)]
pub enum AstroError {
    Ingestion(IngestionError),

    Preprocessing(PreprocessingError),

    Tensor(TensorError),

    Federated(FederatedError),

    Sync(SyncError),

    Config(String),

    Io(io::Error),

    Network(String),

    Serialization(String),

    Database(String),

    InvalidData(String),

    NotFound(String),

    Timeout(String),

    PermissionDenied(String),

    Other(String),
}

#[derive(Debug)]
pub enum IngestionError {
    SourceUnavailable(String),

    InvalidFormat(String),

    DownloadFailed(String),

    ChecksumMismatch,

    RateLimitExceeded,

    AuthenticationFailed,
}

#[derive(Debug)]
pub enum PreprocessingError {
    MissingField(String),

    InvalidValue { field: String, value: String },

    NormalizationFailed(String),

    FeatureExtractionFailed(String),

    OutlierDetectionFailed,
}

#[derive(Debug)]
pub enum TensorError {
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    InvalidDimensions(String),

    EncodingFailed(String),

    BatchCreationFailed(String),

    TypeConversionFailed,
}

#[derive(Debug)]
pub enum FederatedError {
    NodeNotFound(String),

    AggregationFailed(String),

    WeightUpdateFailed(String),

    CommunicationFailed(String),

    IncompatibleVersion { expected: String, got: String },
}

#[derive(Debug)]
pub enum SyncError {
    LockFailed,

    Conflict(String),

    Timeout,

    InconsistentState(String),
}

impl fmt::Display for AstroError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ingestion(e) => write!(f, "Ingestion error: {}", e),
            Self::Preprocessing(e) => write!(f, "Preprocessing error: {}", e),
            Self::Tensor(e) => write!(f, "Tensor error: {}", e),
            Self::Federated(e) => write!(f, "Federated learning error: {}", e),
            Self::Sync(e) => write!(f, "Synchronization error: {}", e),
            Self::Config(msg) => write!(f, "Configuration error: {}", msg),
            Self::Io(e) => write!(f, "I/O error: {}", e),
            Self::Network(msg) => write!(f, "Network error: {}", msg),
            Self::Serialization(msg) => write!(f, "Serialization error: {}", msg),
            Self::Database(msg) => write!(f, "Database error: {}", msg),
            Self::InvalidData(msg) => write!(f, "Invalid data: {}", msg),
            Self::NotFound(msg) => write!(f, "Not found: {}", msg),
            Self::Timeout(msg) => write!(f, "Timeout: {}", msg),
            Self::PermissionDenied(msg) => write!(f, "Permission denied: {}", msg),
            Self::Other(msg) => write!(f, "{}", msg),
        }
    }
}

impl fmt::Display for IngestionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SourceUnavailable(s) => write!(f, "Source unavailable: {}", s),
            Self::InvalidFormat(s) => write!(f, "Invalid format: {}", s),
            Self::DownloadFailed(s) => write!(f, "Download failed: {}", s),
            Self::ChecksumMismatch => write!(f, "Checksum mismatch"),
            Self::RateLimitExceeded => write!(f, "Rate limit exceeded"),
            Self::AuthenticationFailed => write!(f, "Authentication failed"),
        }
    }
}

impl fmt::Display for PreprocessingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingField(field) => write!(f, "Missing required field: {}", field),
            Self::InvalidValue { field, value } => {
                write!(f, "Invalid value '{}' for field '{}'", value, field)
            }
            Self::NormalizationFailed(msg) => write!(f, "Normalization failed: {}", msg),
            Self::FeatureExtractionFailed(msg) => {
                write!(f, "Feature extraction failed: {}", msg)
            }
            Self::OutlierDetectionFailed => write!(f, "Outlier detection failed"),
        }
    }
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeMismatch { expected, got } => {
                write!(f, "Shape mismatch: expected {:?}, got {:?}", expected, got)
            }
            Self::InvalidDimensions(msg) => write!(f, "Invalid dimensions: {}", msg),
            Self::EncodingFailed(msg) => write!(f, "Encoding failed: {}", msg),
            Self::BatchCreationFailed(msg) => write!(f, "Batch creation failed: {}", msg),
            Self::TypeConversionFailed => write!(f, "Type conversion failed"),
        }
    }
}

impl fmt::Display for FederatedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NodeNotFound(id) => write!(f, "Node not found: {}", id),
            Self::AggregationFailed(msg) => write!(f, "Model aggregation failed: {}", msg),
            Self::WeightUpdateFailed(msg) => write!(f, "Weight update failed: {}", msg),
            Self::CommunicationFailed(msg) => write!(f, "Node communication failed: {}", msg),
            Self::IncompatibleVersion { expected, got } => {
                write!(
                    f,
                    "Incompatible version: expected {}, got {}",
                    expected, got
                )
            }
        }
    }
}

impl fmt::Display for SyncError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LockFailed => write!(f, "Failed to acquire lock"),
            Self::Conflict(msg) => write!(f, "Conflict: {}", msg),
            Self::Timeout => write!(f, "Synchronization timeout"),
            Self::InconsistentState(msg) => write!(f, "Inconsistent state: {}", msg),
        }
    }
}

impl std::error::Error for AstroError {}
impl std::error::Error for IngestionError {}
impl std::error::Error for PreprocessingError {}
impl std::error::Error for TensorError {}
impl std::error::Error for FederatedError {}
impl std::error::Error for SyncError {}

impl From<io::Error> for AstroError {
    fn from(err: io::Error) -> Self {
        Self::Io(err)
    }
}

impl From<serde_json::Error> for AstroError {
    fn from(err: serde_json::Error) -> Self {
        Self::Serialization(err.to_string())
    }
}

impl From<csv::Error> for AstroError {
    fn from(err: csv::Error) -> Self {
        Self::Serialization(format!("CSV error: {}", err))
    }
}

impl From<reqwest::Error> for AstroError {
    fn from(err: reqwest::Error) -> Self {
        Self::Network(err.to_string())
    }
}

impl From<IngestionError> for AstroError {
    fn from(err: IngestionError) -> Self {
        Self::Ingestion(err)
    }
}

impl From<PreprocessingError> for AstroError {
    fn from(err: PreprocessingError) -> Self {
        Self::Preprocessing(err)
    }
}

impl From<TensorError> for AstroError {
    fn from(err: TensorError) -> Self {
        Self::Tensor(err)
    }
}

impl From<FederatedError> for AstroError {
    fn from(err: FederatedError) -> Self {
        Self::Federated(err)
    }
}

impl From<SyncError> for AstroError {
    fn from(err: SyncError) -> Self {
        Self::Sync(err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = AstroError::NotFound("test.csv".to_string());
        assert_eq!(err.to_string(), "Not found: test.csv");
    }

    #[test]
    fn test_ingestion_error_conversion() {
        let ing_err = IngestionError::SourceUnavailable("SDSS".to_string());
        let err: AstroError = ing_err.into();
        assert!(matches!(err, AstroError::Ingestion(_)));
    }

    #[test]
    fn test_tensor_shape_mismatch() {
        let err = TensorError::ShapeMismatch {
            expected: vec![10, 5],
            got: vec![10, 3],
        };
        let display = format!("{}", err);
        assert!(display.contains("expected"));
        assert!(display.contains("got"));
    }
}
