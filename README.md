# Mineral Spectral ML Pipeline

**Team Project Umbrella** — Mineral characterisation from ASD spectrometer
reflectance data using machine learning.

## Overview

* Dataset: USGS Spectral Library ASD subset
* 1,276 spectra · 212 classes · 2,151 wavelength bands (0.35–2.5 µm)
* Best model: XGBoost, **87.6% CV accuracy**, 105 classes
* Flask inference server with REST API included

## Models

| File | Description |
|------|-------------|
| `mineral_classifier.pkl` | XGBoost multi-class classifier (105 classes) |
| `abundance_regressor.pkl` | GradientBoosting abundance estimator (%) |
| `feature_scaler.pkl` | StandardScaler fitted on training data |

## Quick start

```bash
git clone https://github.com/norbertmuzila/mineral-spectral-ml.git
cd mineral-spectral-ml
pip install -r requirements.txt
python inference_server.py      # → http://localhost:5050
```

See `mineral_classification.ipynb` for the full ML walkthrough.
