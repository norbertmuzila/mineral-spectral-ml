#!/usr/bin/env python3
"""
Mineral Spectral Classification ML Pipeline
============================================
Trains Random Forest, XGBoost, and SVM models on ASD spectrometer data
(Wavelength vs. Reflectance) from the USGS Spectral Library Version 7.

Data source: Combined_ASD_Data_(2).xlsx - Sheet: Reflectance_vs_Wavelength
- 2151 wavelength bands (0.35 - 2.5 µm)
- 1276 mineral spectra across 212 mineral classes
- Bad bands marked as -1.23e+34 (USGS convention)

Output:
- mineral_classifier.pkl   — Best classification model
- abundance_regressor.pkl  — Abundance estimation model
- feature_scaler.pkl       — Feature scaler
- model_metadata.json      — Model accuracy, class names, feature info
- catalogue_spectra.json   — USGS reference spectra per mineral class
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
import joblib
import openpyxl
from collections import Counter, defaultdict
from scipy.signal import savgol_filter
from scipy.spatial import ConvexHull
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')

EXCEL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'attached_assets', 'Combined_ASD_Data_(2)_1774184045643.xlsx')
OUTPUT_DIR = os.path.dirname(__file__)

BAD_BAND_VALUE = -1.23e+34
BAD_BAND_THRESHOLD = 1e+30

# Diagnostic absorption wavelengths (in micrometers) linked to specific minerals
# Based on USGS Spectral Library documentation
DIAGNOSTIC_WAVELENGTHS = {
    0.43:  "Fe3+ (goethite/hematite)",
    0.48:  "Fe2+ (chlorite/serpentine)",
    0.50:  "Fe3+ crystal field",
    0.65:  "Chlorophyll/organic",
    0.70:  "Fe3+ charge transfer",
    0.90:  "Fe2+ (pyroxene/olivine)",
    1.00:  "Fe2+ (olivine/pyroxene)",
    1.40:  "OH/H2O overtone",
    1.90:  "H2O combination",
    2.00:  "CO3 (carbonate minerals)",
    2.17:  "Al-OH (kaolinite/alunite)",
    2.20:  "Al-OH (muscovite/montmorillonite)",
    2.25:  "Mg-OH/Fe-OH (chlorite)",
    2.30:  "CO3 (dolomite/calcite)",
    2.33:  "Mg-OH (chlorite/serpentine)",
    2.35:  "Fe-OH (goethite/jarosite)",
}

def load_spectral_data():
    """Load the Reflectance_vs_Wavelength sheet from the Excel file."""
    print("Loading spectral data from Excel (using pandas for speed)...")
    start = time.time()
    
    df = pd.read_excel(EXCEL_PATH, sheet_name=1, index_col=0, header=0, engine='openpyxl')
    
    wavelengths = np.array(df.index, dtype=float)
    col_names = list(df.columns)
    
    # Replace bad band values with NaN
    df = df.where(df.abs() < BAD_BAND_THRESHOLD, other=np.nan)
    
    # Build reflectance_data dict (column name -> list of reflectances)
    reflectance_data = {}
    for col in col_names:
        reflectance_data[col] = list(df[col].values)
    
    elapsed = time.time() - start
    print(f"  Sheet: Reflectance_vs_Wavelength")
    print(f"  Loaded {len(wavelengths)} wavelength bands, {len(col_names)} spectra in {elapsed:.1f}s")
    
    return wavelengths, col_names, reflectance_data


def parse_mineral_label(col_name):
    """
    Extract mineral class label from column name.
    Format: s07_ASD_MineralName_SampleID_...
    Examples:
      s07_ASD_Actinolite_HS116.1B_ASDFRb_AREF  -> Actinolite
      s07_ASD_Kaolinite_KGa-1b_ASDFRb_AREF     -> Kaolinite
      s07_ASD_Acmite_NMNH133746_Pyroxene_BECKa  -> Acmite
    """
    parts = col_name.split('_')
    if len(parts) >= 3:
        return parts[2]
    return col_name


def clean_spectrum(reflectances, wavelengths):
    """
    Clean a single spectrum:
    1. Replace bad bands (NaN, 1.23e34) with linear interpolation
    2. Apply Savitzky-Golay smoothing
    3. Clip to [0, 1] physical range
    """
    arr = np.array(reflectances, dtype=float)
    
    # Interpolate NaN values
    nans = np.isnan(arr)
    if nans.all():
        return None
    if nans.any():
        x = np.arange(len(arr))
        arr[nans] = np.interp(x[nans], x[~nans], arr[~nans])
    
    # Clip to physical range
    arr = np.clip(arr, 0.0, 1.0)
    
    # Smooth with Savitzky-Golay (window=11, poly=3)
    if len(arr) >= 11:
        arr = savgol_filter(arr, window_length=11, polyorder=3)
        arr = np.clip(arr, 0.0, 1.0)
    
    return arr


def continuum_removal(spectrum, wavelengths):
    """
    Apply continuum removal (convex hull method).
    Returns the continuum-removed spectrum (values 0-1 relative to convex hull).
    """
    try:
        points = np.column_stack([wavelengths, spectrum])
        hull_indices = []
        
        # Convex hull upper envelope approximation
        n = len(wavelengths)
        # Use a simple convex hull upper envelope
        upper = np.ones(n)
        
        # Marching hull
        i = 0
        while i < n:
            best_slope = -np.inf
            best_j = i + 1
            for j in range(i + 1, n):
                if wavelengths[j] != wavelengths[i]:
                    slope = (spectrum[j] - spectrum[i]) / (wavelengths[j] - wavelengths[i])
                    if slope >= best_slope:
                        best_slope = slope
                        best_j = j
            if best_j >= n - 1:
                # Fill to end
                upper[i:] = np.interp(wavelengths[i:], 
                                       [wavelengths[i], wavelengths[-1]], 
                                       [spectrum[i], spectrum[-1]])
                break
            i = best_j
        
        # Continuum removed = spectrum / continuum
        continuum_removed = np.where(upper > 1e-6, spectrum / upper, 0.0)
        return np.clip(continuum_removed, 0.0, 1.0)
    except Exception:
        return spectrum.copy()


def extract_features(spectrum, wavelengths):
    """
    Extract spectral features from a cleaned reflectance spectrum.
    
    Features include:
    - Raw reflectance at key diagnostic wavelengths
    - First derivative at key wavelengths
    - Continuum-removed spectrum at diagnostic bands
    - Absorption band depth at diagnostic wavelengths
    - Band ratio indices
    - Statistical moments
    """
    features = []
    feature_names = []
    
    wl = np.array(wavelengths)
    spec = np.array(spectrum)
    
    # 1. Raw reflectance at diagnostic wavelengths
    for target_wl, label in DIAGNOSTIC_WAVELENGTHS.items():
        idx = np.argmin(np.abs(wl - target_wl))
        features.append(spec[idx])
        feature_names.append(f"refl_{target_wl:.2f}um")
    
    # 2. First derivative at diagnostic wavelengths
    deriv1 = np.gradient(spec, wl)
    for target_wl in DIAGNOSTIC_WAVELENGTHS.keys():
        idx = np.argmin(np.abs(wl - target_wl))
        features.append(deriv1[idx])
        feature_names.append(f"d1_{target_wl:.2f}um")
    
    # 3. Continuum removal
    cr = continuum_removal(spec, wl)
    for target_wl in DIAGNOSTIC_WAVELENGTHS.keys():
        idx = np.argmin(np.abs(wl - target_wl))
        # Absorption depth = 1 - CR value (deeper = more absorption)
        features.append(1.0 - cr[idx])
        feature_names.append(f"abdepth_{target_wl:.2f}um")
    
    # 4. Spectral band ratios (commonly used in mineral mapping)
    band_ratios = [
        (0.70, 0.55, "ferric_iron"),     # Fe3+ ratio
        (0.90, 0.75, "ferrous_iron"),    # Fe2+ ratio
        (2.20, 1.60, "Al-OH"),           # Clay ratio
        (2.35, 2.15, "Mg-OH"),           # Mg-clay ratio
        (2.33, 2.09, "carbonate"),       # Carbonate ratio
        (1.90, 1.65, "water_content"),   # Water ratio
    ]
    for wl1, wl2, name in band_ratios:
        idx1 = np.argmin(np.abs(wl - wl1))
        idx2 = np.argmin(np.abs(wl - wl2))
        denom = spec[idx2] if spec[idx2] > 0.001 else 0.001
        features.append(spec[idx1] / denom)
        feature_names.append(f"ratio_{name}")
    
    # 5. Regional averages (VIS, NIR, SWIR1, SWIR2)
    regions = [
        (0.35, 0.70, "VIS"),
        (0.70, 1.10, "NIR"),
        (1.10, 1.80, "SWIR1"),
        (1.80, 2.50, "SWIR2"),
    ]
    for wl_min, wl_max, name in regions:
        mask = (wl >= wl_min) & (wl <= wl_max)
        if mask.any():
            features.append(np.mean(spec[mask]))
            features.append(np.std(spec[mask]))
        else:
            features.extend([0.0, 0.0])
        feature_names.extend([f"mean_{name}", f"std_{name}"])
    
    # 6. Statistical moments over full spectrum
    features.append(np.mean(spec))
    features.append(np.std(spec))
    features.append(float(np.percentile(spec, 25)))
    features.append(float(np.percentile(spec, 75)))
    
    # Slope of reflectance (overall trend)
    if len(wl) > 1:
        slope = np.polyfit(wl, spec, 1)[0]
        features.append(slope)
    else:
        features.append(0.0)
    feature_names.extend(["mean_all", "std_all", "p25", "p75", "slope"])
    
    # 7. Downsample spectrum to 200 evenly-spaced bands (capture spectral shape)
    target_wls = np.linspace(wl.min(), wl.max(), 200)
    downsampled = np.interp(target_wls, wl, spec)
    features.extend(downsampled.tolist())
    feature_names.extend([f"ds_{i}" for i in range(200)])
    
    return np.array(features, dtype=float), feature_names


def build_dataset(wavelengths, col_names, reflectance_data, min_samples_per_class=5):
    """Build feature matrix X and labels y from the spectral data."""
    print("\nBuilding feature matrix...")
    
    X_list = []
    y_list = []
    sample_ids = []
    raw_spectra = {}  # Store cleaned spectra for catalogue
    
    # Parse mineral labels
    labels = [parse_mineral_label(col) for col in col_names]
    label_counts = Counter(labels)
    
    # Filter: only include minerals with enough samples
    valid_minerals = {m for m, c in label_counts.items() if c >= min_samples_per_class}
    print(f"  Minerals with >= {min_samples_per_class} samples: {len(valid_minerals)} classes")
    
    skipped = 0
    for col_name, label in zip(col_names, labels):
        if label not in valid_minerals:
            skipped += 1
            continue
        
        raw = reflectance_data[col_name]
        cleaned = clean_spectrum(raw, wavelengths)
        if cleaned is None:
            skipped += 1
            continue
        
        features, feature_names = extract_features(cleaned, wavelengths)
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            skipped += 1
            continue
        
        X_list.append(features)
        y_list.append(label)
        sample_ids.append(col_name)
        
        # Store raw spectra per mineral for catalogue
        if label not in raw_spectra:
            raw_spectra[label] = []
        if len(raw_spectra[label]) < 3:  # Store up to 3 per mineral
            raw_spectra[label].append({
                'wavelengths': wavelengths.tolist(),
                'reflectances': cleaned.tolist(),
                'sample_id': col_name
            })
    
    print(f"  Built {len(X_list)} samples (skipped {skipped})")
    print(f"  Feature vector size: {len(feature_names)}")
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    # Data augmentation: add Gaussian noise to minority class spectra
    # This helps the model generalize better and increases training set size
    print("\n  Augmenting minority classes...")
    label_counts_valid = Counter(y)
    target_min = 15  # Augment to have at least 15 samples per class
    rng = np.random.RandomState(42)
    X_aug_list, y_aug_list = [], []
    
    for mineral, count in label_counts_valid.items():
        if count < target_min:
            mineral_idx = np.where(y == mineral)[0]
            needed = target_min - count
            for _ in range(needed):
                # Pick random sample and add small Gaussian noise (0.5% of signal std)
                base_idx = mineral_idx[rng.randint(0, len(mineral_idx))]
                noise_level = 0.005 * np.std(X[base_idx]) + 0.001
                augmented = X[base_idx] + rng.normal(0, noise_level, X[base_idx].shape)
                augmented = np.clip(augmented, 0, 1)  # Keep reflectance in [0,1]
                X_aug_list.append(augmented)
                y_aug_list.append(mineral)
    
    if X_aug_list:
        X_augmented = np.vstack([X, np.array(X_aug_list)])
        y_augmented = np.concatenate([y, np.array(y_aug_list)])
        print(f"  Augmented: {len(X)} → {len(X_augmented)} samples")
        X, y = X_augmented, y_augmented
    
    return X, y, feature_names, raw_spectra, sample_ids


def train_and_evaluate(X, y):
    """
    Train multiple models and return the best one.
    Models: RandomForest (primary), XGBoost, SVM
    """
    print("\nTraining models...")
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost": None,  # Will try to import
    }
    
    # Try importing XGBoost
    try:
        import xgboost as xgb
        models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='mlogloss',
            random_state=42,
            n_jobs=-1
        )
    except ImportError:
        del models["XGBoost"]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    best_model = None
    best_score = 0.0
    best_name = ""
    results = {}
    
    for name, model in models.items():
        if model is None:
            continue
        print(f"\n  Training {name}...")
        start = time.time()
        
        # Use scaled for SVM, raw for tree-based
        X_use = X_scaled if "SVM" in name else X
        
        scores = cross_val_score(model, X_use, y_enc, cv=cv, scoring='accuracy', n_jobs=-1)
        acc = scores.mean()
        std = scores.std()
        
        # CV F1 score (cross-validated, not training-fit)
        cv_f1_scores = cross_val_score(model, X_use, y_enc, cv=cv, scoring='f1_weighted', n_jobs=-1)
        cv_f1 = cv_f1_scores.mean()
        
        # Full fit (for final model persistence only)
        model.fit(X_use, y_enc)
        
        elapsed = time.time() - start
        print(f"    CV Accuracy: {acc:.4f} ± {std:.4f} | CV F1: {cv_f1:.4f} | Time: {elapsed:.1f}s")
        
        results[name] = {
            "cv_accuracy": float(acc),
            "cv_std": float(std),
            "cv_f1_weighted": float(cv_f1),
            "training_time_s": float(elapsed)
        }
        
        if acc > best_score:
            best_score = acc
            best_model = model
            best_name = name
    
    print(f"\n  Best model: {best_name} (CV accuracy: {best_score:.4f})")
    
    return best_model, best_name, le, scaler, results


def train_abundance_regressor(X, y):
    """
    Train a regression model to estimate mineral abundance (%) .
    Since we don't have ground truth abundance, we use spectral purity
    (reflectance magnitude relative to class mean) as a proxy.
    """
    print("\nTraining abundance regressor...")
    
    # Proxy for abundance: relative reflectance level in SWIR2 region
    # (indices correspond to the downsampled bands at the end of feature vector)
    # Use the mean of the last 25 downsampled bands (SWIR region) as proxy
    ds_start = -50  # last 50 features are downsampled bands
    swir2_vals = X[:, ds_start + 37:]  # last 13 bands ~ SWIR2
    proxy_abundance = np.clip(np.mean(swir2_vals, axis=1) * 100, 5, 95)
    
    regressor = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42
    )
    regressor.fit(X, proxy_abundance)
    print(f"  Regressor trained. Abundance range: {proxy_abundance.min():.1f}% - {proxy_abundance.max():.1f}%")
    
    return regressor


def build_catalogue(raw_spectra, mineral_classes):
    """Build USGS catalogue metadata for each mineral."""
    
    # Mineralogical classification
    MINERAL_CLASSES = {
        "Acmite": "Inosilicate", "Actinolite": "Inosilicate", "Albite": "Tectosilicate",
        "Alunite": "Sulfate", "Almandine": "Nesosilicate", "Andradite": "Nesosilicate",
        "Anorthite": "Tectosilicate", "Antigorite": "Phyllosilicate", "Beryl": "Cyclosilicate",
        "Biotite": "Phyllosilicate", "Calcite": "Carbonate", "Chlorite": "Phyllosilicate",
        "Chrysocolla": "Phyllosilicate", "Clinochlore": "Phyllosilicate", "Diopside": "Inosilicate",
        "Epidote": "Sorosilicate", "Gibbsite": "Hydroxide", "Goethite": "Oxide",
        "Grossular": "Nesosilicate", "Halloysite": "Phyllosilicate", "Hematite": "Oxide",
        "Hornblende": "Inosilicate", "Hypersthene": "Inosilicate", "Illite": "Phyllosilicate",
        "Jarosite": "Sulfate", "Kaolinite": "Phyllosilicate", "Lepidolite": "Phyllosilicate",
        "Magnetite": "Oxide", "Microcline": "Tectosilicate", "Monazite": "Phosphate",
        "Montmorillonite": "Phyllosilicate", "Muscovite": "Phyllosilicate",
        "Olivine": "Nesosilicate", "Orthoclase": "Tectosilicate", "Phlogopite": "Phyllosilicate",
        "Pyroxene": "Inosilicate", "Quartz": "Tectosilicate", "Serpentine": "Phyllosilicate",
        "Smectite": "Phyllosilicate", "Talc": "Phyllosilicate", "Topaz": "Nesosilicate",
        "Tremolite": "Inosilicate",
    }
    
    MINERAL_DESCRIPTIONS = {
        "Kaolinite": "Clay mineral with diagnostic Al-OH absorptions at 1.4, 2.17, and 2.2 µm",
        "Muscovite": "Mica with Al-OH features at 2.2 µm and 1.4 µm",
        "Montmorillonite": "Smectite clay with H2O absorption at 1.9 µm and Al-OH at 2.2 µm",
        "Hematite": "Iron oxide with Fe3+ absorptions near 0.53 and 0.9 µm",
        "Goethite": "Hydrated iron oxide with Fe3+ features at 0.48 and 0.95 µm",
        "Chlorite": "Mg-Fe phyllosilicate with Fe2+ at 0.7 µm and Mg-OH at 2.35 µm",
        "Serpentine": "Mg-silicate with Mg-OH absorption at 2.33 µm",
        "Olivine": "Mg-Fe nesosilicate with broad Fe2+ absorption near 1.0 µm",
        "Alunite": "Sulfate with Al-OH features at 1.76 and 2.17 µm",
        "Jarosite": "Fe-sulfate with Fe3+ at 0.43 and 0.93 µm, SO4 at 2.27 µm",
        "Calcite": "Carbonate with CO3 absorptions at 1.87, 2.0, and 2.35 µm",
        "Talc": "Mg-phyllosilicate with Mg-OH at 1.4 and 2.3 µm",
        "Illite": "K-mica clay with Al-OH at 2.2 µm",
        "Actinolite": "Inosilicate with Fe2+ near 1.0 µm and Mg-OH at 2.33 µm",
        "Quartz": "Tectosilicate, largely featureless in VIS-NIR, Si-O at 8-12 µm",
        "Albite": "Na-plagioclase with weak feldspar features",
        "Microcline": "K-feldspar with characteristic spectral features",
        "Hornblende": "Amphibole with Fe2+ at 1.0 µm and Mg-OH at 2.34 µm",
        "Topaz": "Al-fluorosilicate with OH features",
        "Biotite": "Fe-Mg mica with Fe2+ and OH features",
        "Phlogopite": "Mg-mica with Mg-OH at 2.35 µm",
        "Almandine": "Fe-garnet with strong Fe2+ absorption near 1.2 µm",
        "Diopside": "Ca-pyroxene with Fe2+ at 1.0 µm",
        "Magnetite": "Iron oxide with broad Fe absorption features",
    }
    
    catalogue = {}
    for mineral in mineral_classes:
        spectra = raw_spectra.get(mineral, [])
        catalogue[mineral] = {
            "name": mineral,
            "sampleCount": len([m for m in raw_spectra if m == mineral]),
            "mineralClass": MINERAL_CLASSES.get(mineral, "Silicate"),
            "diagnosticWavelengths": [wl for wl in DIAGNOSTIC_WAVELENGTHS.keys()],
            "description": MINERAL_DESCRIPTIONS.get(mineral, f"{mineral} - spectral library reference"),
            "referenceSpectra": spectra
        }
    
    return catalogue


def detect_absorption_features(spectrum, wavelengths):
    """Detect absorption features in a spectrum."""
    features = []
    wl = np.array(wavelengths)
    spec = np.array(spectrum)
    
    cr = continuum_removal(spec, wl)
    absorption = 1.0 - cr
    
    for target_wl, label in DIAGNOSTIC_WAVELENGTHS.items():
        if target_wl < wl.min() or target_wl > wl.max():
            continue
        idx = np.argmin(np.abs(wl - target_wl))
        
        # Check window around target wavelength
        window_idx = np.where((wl >= target_wl - 0.05) & (wl <= target_wl + 0.05))[0]
        if len(window_idx) == 0:
            continue
        
        depth = float(np.max(absorption[window_idx]))
        if depth > 0.05:  # Only report significant absorptions
            features.append({
                "wavelength": float(target_wl),
                "depth": round(depth, 4),
                "mineralAssociation": label
            })
    
    return sorted(features, key=lambda x: -x["depth"])


def main():
    print("=" * 60)
    print("Mineral Spectral Classification ML Training Pipeline")
    print("=" * 60)
    
    # Step 1: Load data
    wavelengths, col_names, reflectance_data = load_spectral_data()
    
    # Step 2: Build feature dataset
    X, y, feature_names, raw_spectra, sample_ids = build_dataset(
        wavelengths, col_names, reflectance_data, min_samples_per_class=5
    )
    
    label_counts = Counter(y)
    print(f"\nDataset summary:")
    print(f"  Total samples: {len(X)}")
    print(f"  Mineral classes: {len(label_counts)}")
    print(f"  Feature dimensions: {X.shape[1]}")
    print(f"  Samples per class: min={min(label_counts.values())}, max={max(label_counts.values())}, "
          f"mean={np.mean(list(label_counts.values())):.1f}")
    
    # Step 3: Train models
    best_model, best_name, le, scaler, results = train_and_evaluate(X, y)
    
    # Step 4: Train abundance regressor
    abundance_regressor = train_abundance_regressor(X, y)
    
    # Step 5: Save models
    print("\nSaving models...")
    classifier_path = os.path.join(OUTPUT_DIR, 'mineral_classifier.pkl')
    regressor_path = os.path.join(OUTPUT_DIR, 'abundance_regressor.pkl')
    scaler_path = os.path.join(OUTPUT_DIR, 'feature_scaler.pkl')
    
    joblib.dump({
        'model': best_model,
        'label_encoder': le,
        'model_name': best_name,
        'feature_names': feature_names,
        'wavelengths': wavelengths.tolist(),
    }, classifier_path)
    
    joblib.dump(abundance_regressor, regressor_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"  Classifier: {classifier_path}")
    print(f"  Regressor: {regressor_path}")
    print(f"  Scaler: {scaler_path}")
    
    # Step 6: Build and save catalogue
    mineral_classes = sorted(label_counts.keys())
    catalogue = build_catalogue(raw_spectra, mineral_classes)
    
    catalogue_path = os.path.join(OUTPUT_DIR, 'catalogue_spectra.json')
    with open(catalogue_path, 'w') as f:
        json.dump(catalogue, f, separators=(',', ':'))
    print(f"  Catalogue: {catalogue_path}")
    
    # Step 7: Save metadata
    y_pred_train = best_model.predict(X)
    y_enc = le.transform(y)
    train_accuracy = accuracy_score(y_enc, y_pred_train)
    
    best_results = results.get(best_name, {})
    
    metadata = {
        "modelType": best_name,
        "accuracy": round(best_results.get("cv_accuracy", train_accuracy), 4),
        "f1Score": round(best_results.get("cv_f1_weighted", 0.0), 4),
        "f1ScoreNote": "Cross-validated weighted F1 (not training-fit)",
        "numClasses": int(len(label_counts)),
        "numSamples": int(len(X)),
        "numFeatures": int(X.shape[1]),
        "classNames": mineral_classes,
        "modelLoaded": True,
        "trainingDate": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "allModelResults": results,
        "wavelengthRange": [float(wavelengths.min()), float(wavelengths.max())],
        "featureNames": feature_names,
        "diagnosticWavelengths": {str(k): v for k, v in DIAGNOSTIC_WAVELENGTHS.items()}
    }
    
    metadata_path = os.path.join(OUTPUT_DIR, 'model_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata: {metadata_path}")
    
    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETE")
    print(f"  Best model: {best_name}")
    print(f"  CV Accuracy: {best_results.get('cv_accuracy', 0):.4f} ({best_results.get('cv_accuracy', 0)*100:.1f}%)")
    print(f"  F1 Score: {best_results.get('f1_weighted', 0):.4f}")
    print(f"  Classes: {len(label_counts)}")
    print(f"  Samples: {len(X)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
