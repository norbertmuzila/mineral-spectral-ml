# Mineral Spectral ML Pipeline

**Team Project Umbrella** — Mineral characterisation from ASD spectrometer  
reflectance data using machine learning.

## 🌐 Live Demo

**[View the live interactive demo →](https://norbertmuzila.github.io/mineral-spectral-ml/)**

Select from 8 reference mineral spectra, run the XGBoost classifier, and see  
real-time prediction results with absorption feature detection.

## Overview

* Dataset: USGS Spectral Library ASD subset
* 1,276 spectra · 212 classes · 2,151 wavelength bands (0.35–2.5 µm)
* Best model: XGBoost, **87.6% CV accuracy**, 105 classes
* Flask inference server with REST API included
* Interactive web demo (`docs/index.html`)

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
# Open docs/index.html in browser
```

See `mineral_classification.ipynb` for the full ML walkthrough.
