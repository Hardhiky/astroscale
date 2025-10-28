
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataSource {
    Gaia,
    SDSS,
    APOGEE,
    HST,
    JWST,
    RAVE,
}

#[derive(Debug, Clone)]
pub struct AstroData {
    pub source: DataSource,
    pub data: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct DataBatch {
    pub records: Vec<AstroData>,
    pub size: usize,
}
