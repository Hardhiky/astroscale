# AstroScale Core - Architecture Documentation

## Overview

The **AstroScale Node Engine** is a high-performance Rust-based distributed system for astronomical data processing, federated machine learning, and real-time inference serving.

## System Architecture

```

                        AstroScale Core Engine                        

                                                                       
                    
     Ingestion     Preprocessing      Tensor                  
      Layer        Pipeline      Encoding                 
                    
                                                                   
                               
                                                                     
                                                      
                     Federated                                      
                      Learning                                      
                     Coordinator                                    
                                                      
                                                                     
                               
                                                                   
                                
    Node 1    Node 2    Node 3                    
   (Worker)        (Worker)        (Worker)                   
                                
                                                                       

                               
                               
                    
                      OCaml Backend API 
                        (Port 8080)     
                    
                               
                               
                    
                      Svelte Frontend   
                        (Port 5173)     
                    
```

## Core Components

### 1. Data Ingestion Layer

**Purpose**: Fetch and ingest astronomical data from distributed sources.

**Sources Supported**:
- **Gaia DR3**: European Space Agency's astrometry mission
- **SDSS DR17**: Sloan Digital Sky Survey spectroscopic data
- **APOGEE DR17**: High-resolution infrared spectroscopy
- **HST**: Hubble Space Telescope archive
- **JWST**: James Webb Space Telescope data
- **RAVE DR6**: Radial Velocity Experiment

**Features**:
- Concurrent downloads with rate limiting
- Retry logic with exponential backoff
- Checksum verification
- Progress tracking and resumable downloads
- Local caching for efficiency

**Key Files**:
- `src/ingest/mod.rs` - Main ingestion coordinator
- `src/ingest/sources/` - Source-specific adapters
- `src/ingest/cache.rs` - Download caching system

### 2. Preprocessing Pipeline

**Purpose**: Clean, normalize, and prepare data for ML models.

**Operations**:
1. **Data Validation**
   - Schema validation
   - Type checking
   - Range verification

2. **Missing Value Handling**
   - Mean/median imputation
   - Forward/backward fill
   - Intelligent interpolation

3. **Normalization**
   - Standard scaling (z-score)
   - Min-max scaling
   - Robust scaling (IQR-based)

4. **Outlier Detection**
   - Z-score method
   - IQR method
   - Isolation Forest

5. **Feature Engineering**
   - Derived features (distance from parallax)
   - Polynomial features
   - Interaction terms
   - Temporal features

**Parallelization**:
- Uses Rayon for CPU-level parallelism
- Batch processing for memory efficiency
- Configurable worker threads

**Key Files**:
- `src/preprocess/mod.rs` - Pipeline coordinator
- `src/preprocess/normalize.rs` - Normalization strategies
- `src/preprocess/outliers.rs` - Outlier detection
- `src/preprocess/features.rs` - Feature engineering

### 3. Tensor Encoding Layer

**Purpose**: Convert preprocessed data into tensor format for ML models.

**Capabilities**:
- N-dimensional array support via `ndarray`
- Efficient batch creation
- Memory-mapped tensors for large datasets
- Type-safe tensor operations
- Serialization/deserialization

**Tensor Formats**:
- **NPY**: NumPy format for Python interop
- **Bincode**: Rust native format
- **Arrow**: Apache Arrow for columnar data

**Key Files**:
- `src/tensor/mod.rs` - Tensor operations
- `src/tensor/batch.rs` - Batch creation
- `src/tensor/encoder.rs` - Data encoding

### 4. Federated Learning System

**Purpose**: Coordinate distributed model training across multiple nodes.

**Architecture**:

```
Coordinator Node
    
     Worker Node 1 (Local Training)
     Worker Node 2 (Local Training)
     Worker Node 3 (Local Training)
         
          Aggregation  Model Update  Broadcast
```

**Algorithms Supported**:
- **FedAvg**: Federated Averaging (default)
- **FedProx**: Proximal term for heterogeneity
- **FedAdam**: Adaptive optimization
- **Scaffold**: Variance reduction

**Privacy**:
- Differential Privacy (DP) support
- Secure aggregation protocols
- Gradient clipping
- Privacy budget tracking

**Synchronization**:
- Periodic sync intervals
- Event-driven updates
- Conflict resolution
- Version control for models

**Key Files**:
- `src/federated/mod.rs` - Federated coordinator
- `src/federated/aggregation.rs` - Aggregation strategies
- `src/federated/privacy.rs` - Differential privacy
- `src/federated/sync.rs` - Synchronization logic

### 5. Node Management

**Purpose**: Manage node lifecycle, health, and communication.

**Node Types**:
- **Coordinator**: Orchestrates federated learning, aggregates models
- **Worker**: Performs local training, sends updates
- **Hybrid**: Can act as both coordinator and worker

**Health Monitoring**:
- Heartbeat system
- Resource usage tracking
- Performance metrics
- Failure detection

**Communication**:
- REST API for control plane
- gRPC for data plane (optional)
- WebSocket for real-time updates
- Message queuing for async tasks

**Key Files**:
- `src/node/mod.rs` - Node implementation
- `src/node/health.rs` - Health checks
- `src/node/api.rs` - HTTP API

### 6. Metrics & Observability

**Purpose**: Monitor system health and performance.

**Metrics Collected**:
- Data ingestion rate
- Preprocessing throughput
- Model training metrics (loss, accuracy)
- Resource utilization (CPU, RAM, GPU)
- Network I/O
- Error rates

**Export Formats**:
- Prometheus metrics endpoint
- JSON API
- StatsD protocol
- OpenTelemetry (planned)

**Key Files**:
- `src/metrics/mod.rs` - Metrics collection
- `src/metrics/prometheus.rs` - Prometheus exporter

## Data Flow

### 1. Ingestion Flow

```
External Source → HTTP Request → Download → Validate → Cache → Store
                                      ↓
                                 Checksum?
                                      ↓
                                   [Yes/No]
```

### 2. Preprocessing Flow

```
Raw Data → Validate → Clean → Normalize → Feature Engineering → Tensor
    ↓         ↓         ↓          ↓              ↓               ↓
  Schema   Missing   Outliers  Scaling      New Features     ndarray
```

### 3. Federated Training Flow

```
[Coordinator]
    ↓
Broadcast Model
    ↓
[Workers: Local Training]
    ↓
Send Gradients/Weights
    ↓
[Coordinator: Aggregate]
    ↓
Update Global Model
    ↓
Broadcast Updated Model
    ↓
[Repeat]
```

## Configuration

### Configuration File Structure

```toml
[node]
id = "node-1"
name = "AstroScale Worker 1"
role = "worker"
bind_address = "0.0.0.0"
port = 8081

[ingestion]
concurrent_downloads = 4
download_timeout_secs = 300

[[ingestion.sources]]
name = "gaia"
endpoint = "https://gea.esac.esa.int/tap-server/tap"
enabled = true

[preprocessing]
batch_size = 1024
normalization = "standard"
enable_parallel = true

[federated]
enabled = true
coordinator_address = "http://coordinator:8081"
sync_interval_secs = 300
aggregation_strategy = "fedavg"

[storage]
data_dir = "./data"
cache_dir = "./cache"
model_dir = "./models"

[resources]
max_memory_gb = 16.0
max_cpu_cores = 8
enable_gpu = true
```

## API Endpoints

### Node API (Port 8081)

#### `POST /ingest`
Trigger data ingestion from specified sources.

**Request**:
```json
{
  "sources": ["gaia", "sdss"],
  "start_date": "2024-01-01",
  "limit": 10000
}
```

#### `POST /preprocess`
Process raw data through preprocessing pipeline.

**Request**:
```json
{
  "input_path": "./data/raw/sample.csv",
  "output_path": "./data/processed/",
  "batch_size": 1024
}
```

#### `GET /status`
Get node status and health.

**Response**:
```json
{
  "node_id": "node-1",
  "status": "healthy",
  "uptime_secs": 86400,
  "memory_usage_mb": 2048,
  "cpu_usage_percent": 45.2
}
```

#### `GET /metrics`
Get current metrics.

**Response**:
```json
{
  "ingestion_rate": 1250.5,
  "preprocessing_throughput": 8500.0,
  "model_accuracy": 0.85,
  "error_rate": 0.001
}
```

### Federated API

#### `POST /federated/join`
Register as worker node with coordinator.

#### `POST /federated/update`
Submit model update to coordinator.

#### `GET /federated/model`
Fetch latest global model.

## Performance Characteristics

### Throughput

- **Ingestion**: 10,000 records/sec (single source)
- **Preprocessing**: 50,000 records/sec (8-core CPU)
- **Tensor Encoding**: 100,000 records/sec

### Latency

- **API Response**: < 10ms (p99)
- **Preprocessing**: < 100ms per batch (1024 records)
- **Model Sync**: < 5s (federated update)

### Scalability

- **Horizontal**: Add more worker nodes
- **Vertical**: Increase CPU/RAM per node
- **Data**: Handles datasets > 1TB

## Error Handling

### Error Types

1. **Ingestion Errors**: Network failures, authentication, rate limits
2. **Preprocessing Errors**: Invalid data, missing fields
3. **Tensor Errors**: Shape mismatches, type conversions
4. **Federated Errors**: Node failures, version conflicts
5. **System Errors**: OOM, disk full, permission denied

### Recovery Strategies

- **Retry Logic**: Exponential backoff for transient failures
- **Checkpointing**: Save progress for long-running tasks
- **Graceful Degradation**: Continue with partial data
- **Circuit Breaker**: Stop failing operations temporarily

## Security

### Data Security

- TLS/SSL for all network communication
- Data encryption at rest (optional)
- Secure credential storage
- Access control lists

### Privacy

- Differential privacy for federated learning
- Gradient clipping to prevent model inversion
- Secure aggregation protocols
- Data anonymization options

## Testing

### Unit Tests
```bash
cargo test
```

### Integration Tests
```bash
cargo test --test integration
```

### Benchmarks
```bash
cargo bench
```

## Deployment

### Single Node

```bash
cargo build --release
./target/release/astroscale-node --config config.toml
```

### Coordinator + Workers

```bash
# Coordinator
./astroscale-node --config coordinator.toml

# Workers (multiple machines)
./astroscale-node --config worker1.toml
./astroscale-node --config worker2.toml
```

### Docker

```bash
docker build -t astroscale-core .
docker run -p 8081:8081 -v ./data:/data astroscale-core
```

### Kubernetes

```bash
kubectl apply -f k8s/deployment.yaml
```

## Integration with Existing Stack

### OCaml Backend

The Rust core communicates with the OCaml backend via:
- **HTTP API**: RESTful endpoints
- **File System**: Shared data directory
- **Unix Pipes**: For streaming data

### Python ML Models

The Rust core provides:
- **NPY Format**: NumPy-compatible tensors
- **Arrow Format**: Zero-copy data sharing
- **JSON API**: For metadata exchange

## Roadmap

### Phase 1 (Current)
-  Core architecture
-  Configuration system
-  Error handling
-  Data ingestion
-  Preprocessing pipeline

### Phase 2
-  Tensor encoding
-  Federated learning
-  Metrics system

### Phase 3
-  Advanced aggregation algorithms
-  Differential privacy
-  GPU acceleration

### Phase 4
-  ONNX Runtime integration
-  Distributed tracing
-  Auto-scaling

## Contributing

See `CONTRIBUTING.md` for development guidelines.

## License

MIT License - See `LICENSE` file.

---

**Last Updated**: 2024  
**Version**: 0.1.0  
**Status**: Development 