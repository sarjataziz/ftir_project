# FTIR Compound Classification System

This project for chemical compound classification using FTIR (Fourier Transform Infrared) spectroscopy. 
## Overview

The solution combines:

* ✨ **Random Forest + XGBoost Ensemble** (for explainable predictions)
* 🧐 **CNN + BiLSTM + Attention Deep Learning** model (for sequential pattern learning)

It supports complete preprocessing, training, model optimization, evaluation, and compound prediction from raw FTIR spectra.

---

## Project Structure

```
ftir_project/
├── data/                     # Raw and prediction input files
├── models/                   # Saved model artifacts
├── save_point/              # Intermediate files (npy, pkl, json)
├── scripts/                 # Core processing and training scripts
│   ├── preprocessing.py     # Feature engineering and scaling
│   ├── train_rf_xgb.py      # RF + XGB training with BayesSearchCV
│   ├── train_dl_model.py    # DL training using KerasTuner
│   ├── evaluate.py          # Evaluation tools (confusion, ROC)
│   └── predict.py           # Load CSV and predict using trained models
├── main.py                  # Central orchestrator (evaluation entrypoint)
└── README.md                # Documentation
```

---

## Technology Stack

| Domain | Tools & Libraries                                       |
| ------ | ------------------------------------------------------- |
| ML     | `scikit-learn`, `xgboost`, `bayes_opt`, `joblib`        |
| DL     | `tensorflow`, `keras-tuner`, `LSTM`, `CNN`, `Attention` |
| Data   | `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`     |
| Format | `.xlsx`, `.csv`, `.npy`, `.pkl`, `.json`, `.h5`         |

---

## Setup Instructions

### 1. Environment Setup

```bash
pip install -r requirements.txt
```

### 2. Train Traditional Models (RF + XGB)

```bash
python scripts/train_rf_xgb.py
```

### 3. Train Deep Learning Model

```bash
python scripts/train_dl_model.py
```

### 4. Evaluate Models

```bash
python main.py
```

### 5. Predict on Custom Data

```bash
python scripts/predict.py
```

> Place your test file in `data/sample_input.csv`

---

## Input Format

CSV input include:

```
wavenumbers,transmittance
400.0,2.13
402.0,2.22
...
```

The script will automatically convert between `transmittance` and `absorbance` if only one is provided.

---

## Output

| Output File                           | Description                       |
| ------------------------------------- | --------------------------------- |
| `models/rf_xgb_ensemble.pkl`          | Soft-voting model for prediction  |
| `models/best_model.h5`                | Final deep learning model         |
| `save_point/*.pkl`, `*.json`, `*.npy` | Encoders, scalers, dataset splits |

---

## Key Features

* ✅ Reproducible training pipelines
* ✅ Modular code (clean separation of concerns)
* ✅ Preprocessing for both ML and DL
* ✅ Custom attention layer for interpretability
* ✅ Auto-handling of real-world FTIR spectral inputs
* ✅ Accepts both `transmittance` and `absorbance` columns

---

