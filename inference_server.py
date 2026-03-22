#!/usr/bin/env python3
"""
Mineral Spectral Inference Server (Flask)
==========================================
Provides mineral classification inference using the trained ML model.
Called from the Express API server as a subprocess/HTTP sidecar.
Runs on port 5050 internally.
"""

import os
import sys
import json
import time
import numpy as np
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

MODEL_DIR = os.path.dirname(__file__)

app = Flask(__name__)
CORS(app)

# Global model state
classifier_bundle = None
abundance_regressor = None
scaler = None
model_metadata = None
catalogue = None

BAD_BAND_THRESHOLD = 1e+30

DIAGNOSTIC_WAVELENGTHS = {
    0.43: "Fe3+ (goethite/hematite)",
    0.48: "Fe2+ (chlorite/serpentine)",
    0.50: "Fe3+ crystal field",
    0.65: "Chlorophyll/organic",
    0.70: "Fe3+ charge transfer",
    0.90: "Fe2+ (pyroxene/olivine)",
    1.00: "Fe2+ (olivine/pyroxene)",
    1.40: "OH/H2O overtone",
    1.90: "H2O combination",
    2.00: "CO3 (carbonate minerals)",
    2.17: "Al-OH (kaolinite/alunite)",
    2.20: "Al-OH (muscovite/montmorillonite)",
    2.25: "Mg-OH/Fe-OH (chlorite)",
    2.30: "CO3 (dolomite/calcite)",
    2.33: "Mg-OH (chlorite/serpentine)",
    2.35: "Fe-OH (goethite/jarosite)",
}


def load_models():
    """Load all model artifacts from disk."""
    global classifier_bundle, abundance_regressor, scaler, model_metadata, catalogue

    classifier_path = os.path.join(MODEL_DIR, 'mineral_classifier.pkl')
    regressor_path = os.path.join(MODEL_DIR, 'abundance_regressor.pkl')
    scaler_path = os.path.join(MODEL_DIR, 'feature_scaler.pkl')
    metadata_path = os.path.join(MODEL_DIR, 'model_metadata.json')
    catalogue_path = os.path.join(MODEL_DIR, 'catalogue_spectra.json')

    if not os.path.exists(classifier_path):
        print(f"[inference_server] Model not found at {classifier_path}", flush=True)
        print("[inference_server] Run train_model.py first to train the model", flush=True)
        return False

    try:
        print("[inference_server] Loading classifier...", flush=True)
        classifier_bundle = joblib.load(classifier_path)
        print("[inference_server] Loading regressor...", flush=True)
        abundance_regressor = joblib.load(regressor_path)
        print("[inference_server] Loading scaler...", flush=True)
        scaler = joblib.load(scaler_path)

        with open(metadata_path) as f:
            model_metadata = json.load(f)

        with open(catalogue_path) as f:
            catalogue = json.load(f)

        print(f"[inference_server] Models loaded. Classes: {model_metadata['numClasses']}", flush=True)
        return True

    except Exception as e:
        print(f"[inference_server] Error loading models: {e}", flush=True)
        return False


def clean_spectrum(wavelengths, reflectances):
    """Clean and validate a spectrum."""
    from scipy.signal import savgol_filter

    wl = np.array(wavelengths, dtype=float)
    spec = np.array(reflectances, dtype=float)

    # Replace bad bands
    bad = np.isnan(spec) | np.isinf(spec) | (np.abs(spec) > BAD_BAND_THRESHOLD)
    if bad.all():
        return None, None
    if bad.any():
        x = np.arange(len(spec))
        spec[bad] = np.interp(x[bad], x[~bad], spec[~bad])

    # Clip to physical range
    spec = np.clip(spec, 0.0, 1.0)

    # Smooth
    if len(spec) >= 11:
        try:
            spec = savgol_filter(spec, window_length=11, polyorder=3)
            spec = np.clip(spec, 0.0, 1.0)
        except Exception:
            pass

    return wl, spec


def continuum_removal(spectrum, wavelengths):
    """Apply continuum removal."""
    try:
        wl = np.array(wavelengths)
        spec = np.array(spectrum)
        n = len(wl)
        upper = np.ones(n)
        i = 0
        while i < n:
            best_slope = -np.inf
            best_j = i + 1
            for j in range(i + 1, n):
                if wl[j] != wl[i]:
                    slope = (spec[j] - spec[i]) / (wl[j] - wl[i])
                    if slope >= best_slope:
                        best_slope = slope
                        best_j = j
            if best_j >= n - 1:
                upper[i:] = np.interp(wl[i:], [wl[i], wl[-1]], [spec[i], spec[-1]])
                break
            i = best_j

        cr = np.where(upper > 1e-6, spec / upper, 0.0)
        return np.clip(cr, 0.0, 1.0)
    except Exception:
        return np.array(spectrum)


def extract_features(spectrum, wavelengths):
    """Extract features from spectrum (must match train_model.py)."""
    wl = np.array(wavelengths)
    spec = np.array(spectrum)
    features = []

    # 1. Raw reflectance at diagnostic wavelengths
    for target_wl in DIAGNOSTIC_WAVELENGTHS.keys():
        idx = np.argmin(np.abs(wl - target_wl))
        features.append(spec[idx])

    # 2. First derivative
    deriv1 = np.gradient(spec, wl)
    for target_wl in DIAGNOSTIC_WAVELENGTHS.keys():
        idx = np.argmin(np.abs(wl - target_wl))
        features.append(deriv1[idx])

    # 3. Continuum removal absorption depths
    cr = continuum_removal(spec, wl)
    for target_wl in DIAGNOSTIC_WAVELENGTHS.keys():
        idx = np.argmin(np.abs(wl - target_wl))
        features.append(1.0 - cr[idx])

    # 4. Band ratios
    band_ratios = [
        (0.70, 0.55), (0.90, 0.75), (2.20, 1.60),
        (2.35, 2.15), (2.33, 2.09), (1.90, 1.65),
    ]
    for wl1, wl2 in band_ratios:
        idx1 = np.argmin(np.abs(wl - wl1))
        idx2 = np.argmin(np.abs(wl - wl2))
        denom = spec[idx2] if spec[idx2] > 0.001 else 0.001
        features.append(spec[idx1] / denom)

    # 5. Regional averages
    regions = [(0.35, 0.70), (0.70, 1.10), (1.10, 1.80), (1.80, 2.50)]
    for wl_min, wl_max in regions:
        mask = (wl >= wl_min) & (wl <= wl_max)
        if mask.any():
            features.append(np.mean(spec[mask]))
            features.append(np.std(spec[mask]))
        else:
            features.extend([0.0, 0.0])

    # 6. Statistical moments
    features.append(np.mean(spec))
    features.append(np.std(spec))
    features.append(float(np.percentile(spec, 25)))
    features.append(float(np.percentile(spec, 75)))

    if len(wl) > 1:
        slope = np.polyfit(wl, spec, 1)[0]
        features.append(slope)
    else:
        features.append(0.0)

    # 7. Downsampled spectrum
    target_wls = np.linspace(wl.min(), wl.max(), 200)
    downsampled = np.interp(target_wls, wl, spec)
    features.extend(downsampled.tolist())

    return np.array(features, dtype=float)


def detect_absorption_features(spectrum, wavelengths):
    """Detect significant absorption features."""
    wl = np.array(wavelengths)
    spec = np.array(spectrum)
    cr = continuum_removal(spec, wl)
    absorption = 1.0 - cr
    features = []

    for target_wl, label in DIAGNOSTIC_WAVELENGTHS.items():
        if target_wl < wl.min() or target_wl > wl.max():
            continue
        window_idx = np.where((wl >= target_wl - 0.05) & (wl <= target_wl + 0.05))[0]
        if len(window_idx) == 0:
            continue
        depth = float(np.max(absorption[window_idx]))
        if depth > 0.05:
            features.append({
                "wavelength": float(target_wl),
                "depth": round(depth, 4),
                "mineralAssociation": label
            })

    return sorted(features, key=lambda x: -x["depth"])


@app.route('/healthz', methods=['GET'])
def health():
    return jsonify({"status": "ok", "modelLoaded": classifier_bundle is not None})


@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint."""
    if classifier_bundle is None:
        return jsonify({"error": "Model not loaded. Run train_model.py first."}), 503

    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400

    wavelengths = data.get('wavelengths')
    reflectances = data.get('reflectances')

    if not wavelengths or not reflectances:
        return jsonify({"error": "wavelengths and reflectances are required"}), 400

    if len(wavelengths) != len(reflectances):
        return jsonify({"error": "wavelengths and reflectances must have equal length"}), 400

    if len(wavelengths) < 10:
        return jsonify({"error": "At least 10 data points required"}), 400

    try:
        wl, spec = clean_spectrum(wavelengths, reflectances)
        if wl is None:
            return jsonify({"error": "All reflectance values are invalid (bad bands)"}), 400

        features = extract_features(spec, wl)
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            return jsonify({"error": "Feature extraction produced invalid values"}), 400

        X = features.reshape(1, -1)

        model = classifier_bundle['model']
        le = classifier_bundle['label_encoder']

        # Predict class and probability
        y_pred = model.predict(X)
        predicted_class = le.inverse_transform(y_pred)[0]
        confidence = 0.0
        top_candidates = []

        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)[0]
            confidence = float(np.max(proba))
            top_idx = np.argsort(proba)[::-1][:5]
            top_candidates = [
                {"mineralClass": le.inverse_transform([i])[0], "confidence": round(float(proba[i]), 4)}
                for i in top_idx if proba[i] > 0.001
            ]
        else:
            confidence = 0.85
            top_candidates = [{"mineralClass": predicted_class, "confidence": confidence}]

        # Abundance estimate
        abundance = float(np.clip(abundance_regressor.predict(X)[0], 0, 100))

        # Absorption features
        absorption_features = detect_absorption_features(spec.tolist(), wl.tolist())

        # Find USGS reference match
        usgs_match = predicted_class if predicted_class in catalogue else None

        return jsonify({
            "mineralClass": predicted_class,
            "confidence": round(confidence, 4),
            "abundanceEstimate": round(abundance, 1),
            "absorptionFeatures": absorption_features[:8],
            "topCandidates": top_candidates,
            "usgsReferenceMatch": usgs_match,
            "modelUsed": classifier_bundle.get('model_name', 'RandomForest')
        })

    except Exception as e:
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500


@app.route('/model-info', methods=['GET'])
def model_info():
    """Return model metadata."""
    if model_metadata is None:
        return jsonify({
            "modelType": "RandomForest",
            "accuracy": 0.0,
            "f1Score": 0.0,
            "numClasses": 0,
            "numSamples": 0,
            "numFeatures": 0,
            "classNames": [],
            "modelLoaded": False,
            "trainingDate": None
        })
    return jsonify(model_metadata)


@app.route('/catalogue', methods=['GET'])
def get_catalogue():
    """Return mineral catalogue."""
    if catalogue is None:
        return jsonify({"minerals": [], "totalCount": 0})

    minerals = []
    for name, info in catalogue.items():
        minerals.append({
            "name": info["name"],
            "sampleCount": info.get("sampleCount", 0),
            "mineralClass": info.get("mineralClass", "Unknown"),
            "diagnosticWavelengths": info.get("diagnosticWavelengths", []),
            "description": info.get("description", "")
        })

    minerals.sort(key=lambda x: x["name"])
    return jsonify({"minerals": minerals, "totalCount": len(minerals)})


@app.route('/catalogue/<mineral_name>', methods=['GET'])
def get_mineral_spectrum(mineral_name):
    """Return reference spectrum for a specific mineral."""
    if catalogue is None:
        return jsonify({"error": "Catalogue not loaded"}), 503

    # Case-insensitive search
    found = None
    for name, info in catalogue.items():
        if name.lower() == mineral_name.lower():
            found = info
            break

    if found is None:
        return jsonify({"error": f"Mineral '{mineral_name}' not found in catalogue"}), 404

    spectra = found.get("referenceSpectra", [])
    if not spectra:
        return jsonify({"error": f"No reference spectrum available for '{mineral_name}'"}), 404

    # Return first reference spectrum
    ref = spectra[0]
    return jsonify({
        "name": found["name"],
        "wavelengths": ref["wavelengths"],
        "reflectances": ref["reflectances"],
        "sampleId": ref.get("sample_id", ""),
        "mineralClass": found.get("mineralClass", "Unknown")
    })


@app.route('/reload', methods=['POST'])
def reload_models():
    """Hot-reload models from disk (call after training completes)."""
    success = load_models()
    if success:
        return jsonify({"status": "reloaded", "modelLoaded": True})
    return jsonify({"status": "failed", "modelLoaded": False}), 503


if __name__ == '__main__':
    port = int(os.environ.get('ML_SERVER_PORT', 5050))
    print(f"[inference_server] Starting on port {port}", flush=True)
    loaded = load_models()
    if not loaded:
        print("[inference_server] WARNING: Models not loaded. Train first.", flush=True)
    app.run(host='0.0.0.0', port=port, debug=False)
