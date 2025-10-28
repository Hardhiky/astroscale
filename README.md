# AstroScale - Stellar Redshift Prediction System

A production-ready machine learning system for predicting stellar redshift from spectroscopic parameters.

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![ML Model](https://img.shields.io/badge/model-gradient_boosting-green)
![Accuracy](https://img.shields.io/badge/R²-0.30-orange)

##  Overview

AstroScale predicts stellar redshift (z) using machine learning trained on 19,000 stellar observations. The system combines spectroscopic parameters (temperature, gravity, metallicity) with distance indicators to estimate redshift values.

### Key Features

- **Machine Learning Prediction**: Gradient Boosting model with R² = 0.30
- **Modern Web Interface**: Beautiful, responsive UI with real-time visualization
- **Fast Backend**: OCaml server with CORS support
- **Production Ready**: Clean architecture, error handling, logging

##  Architecture

```

   Frontend        Svelte + TailwindCSS
   (Port 5173)     Modern UI with visualizations

          HTTP/JSON
         

   Backend         OCaml + Cohttp
   (Port 8080)     API server with CORS

          Process call
         

   ML Model        Python + scikit-learn
   inference.py    Gradient Boosting

```

##  Quick Start

### Prerequisites

- **OCaml** (5.3.0+)
- **Dune** (3.20+)
- **Node.js** (18+)
- **Python** (3.11+) with virtualenv

### Installation

1. **Clone and navigate**
```bash
cd projects/astroscale
```

2. **Setup Python environment**
```bash
# Activate virtualenv (adjust path if different)
source ~/.virtualenvs/py3.11/bin/activate

# Install Python dependencies
pip install numpy scikit-learn pickle
```

3. **Install frontend dependencies**
```bash
cd frontend
npm install
cd ..
```

4. **Build backend**
```bash
cd backend
dune build
cd ..
```

### Running the Application

**Terminal 1 - Backend Server:**
```bash
cd backend
dune exec backend
```

**Terminal 2 - Frontend Dev Server:**
```bash
cd frontend
npm run dev
```

Open your browser to `http://localhost:5173`

##  Model Information

### Training Data
- **Sources**: APOGEE DR17, SDSS DR17, RAVE DR6
- **Samples**: 19,000 stellar observations
- **Features**: RA, Dec, Teff, log g, [Fe/H], SNR, Parallax

### Model Performance
- **Algorithm**: Gradient Boosting Regressor
- **Training R²**: 0.66
- **Validation R²**: 0.30
- **RMSE**: 0.000476

### Input Parameters

| Parameter | Unit | Range | Description |
|-----------|------|-------|-------------|
| `ra` | degrees | 0-360 | Right Ascension |
| `dec` | degrees | -90 to 90 | Declination |
| `teff` | Kelvin | 2000-50000 | Effective Temperature |
| `logg` | log(cm/s²) | 0-5 | Surface Gravity |
| `fe_h` | dex | -3 to 0.5 | Metallicity |
| `snr` | - | >0 | Signal-to-Noise Ratio |
| `parallax` | mas | 0.01-100 | Parallax |

##  Development

### Project Structure

```
astroscale/
 backend/                 # OCaml API server
    bin/main.ml         # Server implementation
    lib/                # Library code
    dune-project        # Dune configuration
 frontend/               # Svelte web application
    src/routes/         # Pages and routes
    static/             # Static assets
    package.json        # Dependencies
 datasets/               # Training data
    catalogs/           # Source catalogs
    ml_preprocessed/    # Processed data & models
 scripts/                # Training pipeline
    prepare_training_data.py
    train_best_model.py
    export_gb_model.py
 inference_production.py # ML inference script
```

### Retraining the Model

```bash
cd scripts

# 1. Prepare training data
~/.virtualenvs/py3.11/bin/python prepare_training_data.py

# 2. Train models and select best
~/.virtualenvs/py3.11/bin/python train_best_model.py

# 3. Export production model
~/.virtualenvs/py3.11/bin/python export_gb_model.py
```

### API Endpoints

#### `POST /predict`

Predict stellar redshift from parameters.

**Request:**
```json
{
  "ra": 200.12,
  "dec": -47.33,
  "teff": 5800,
  "logg": 4.3,
  "fe_h": 0.0,
  "snr": 100,
  "parallax": 7.2
}
```

**Response:**
```json
{
  "predicted_z": 0.000254
}
```

**CORS**: Enabled for all origins

##  Frontend Features

### Visualizations

1. **Redshift Spectrum**: Log-scale visualization showing prediction position
2. **Star Visualization**: Color-coded by temperature (2000K-50000K)
3. **Prediction History**: Track recent predictions
4. **Distance Interpretation**: Automatic categorization of results

### Preset Stellar Types

- **Sun-like Star**: G-type main sequence (5800K)
- **Hot Blue Star**: A-type star (8500K)
- **Red Giant**: Evolved star (4200K)
- **Metal-poor Star**: Population II star
- **White Dwarf**: Dense stellar remnant (12000K)

##  Testing

Test the inference directly:

```bash
~/.virtualenvs/py3.11/bin/python inference_production.py \
  200.12 -47.33 5800 4.3 0.0 100 7.2
```

Expected output: `0.000254`

##  Model Interpretation

### Redshift Categories

| z Range | Category | Distance | Example |
|---------|----------|----------|---------|
| < 0.0001 | Very Nearby | < 100 pc | Local stars |
| 0.0001-0.001 | Nearby | 100-1000 pc | Milky Way |
| 0.001-0.01 | Distant | 1-10 kpc | Galactic disk |
| 0.01-0.1 | Very Distant | > 10 kpc | Nearby galaxies |
| > 0.1 | Extremely Distant | > 100 kpc | Galaxy clusters |

### Physical Relationships

The model learns correlations between:
- **Parallax** → Distance → Redshift (primary)
- **Metallicity** → Stellar age → Population
- **Temperature + Gravity** → Stellar type → Intrinsic brightness
- **Position** → Galactic structure

##  Troubleshooting

### Backend won't start

Check OCaml dependencies:
```bash
cd backend
opam install . --deps-only
dune build
```

### Frontend build errors

Clear cache and reinstall:
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```

### Python import errors

Ensure virtualenv is activated:
```bash
source ~/.virtualenvs/py3.11/bin/activate
pip install numpy scikit-learn
```

### Model not found

Export the production model:
```bash
~/.virtualenvs/py3.11/bin/python scripts/export_gb_model.py
```

##  License

MIT License - See LICENSE file for details

##  Contributors

Built with  for astronomical research

##  References

- **APOGEE**: Apache Point Observatory Galactic Evolution Experiment
- **SDSS**: Sloan Digital Sky Survey
- **RAVE**: RAdial Velocity Experiment
- **Gaia**: European Space Agency astrometry mission

---

**Version**: 1.0.0  
**Last Updated**: 2024  
**Status**: Production Ready 