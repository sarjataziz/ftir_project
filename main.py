import os
import numpy as np
import joblib
from scripts.train_rf_xgb import *
from scripts.train_dl_model import *
from scripts.evaluate import evaluate_model

SAVE_PATH = os.path.join("save_point")
MODEL_PATH = os.path.join("models")

# Load saved data
X_test = np.load(os.path.join(SAVE_PATH, "X_test.npy"))
y_test = np.load(os.path.join(SAVE_PATH, "y_test.npy"))

# Load RF/XGB Ensemble Model
ensemble = joblib.load(os.path.join(MODEL_PATH, "rf_xgb_ensemble.pkl"))
print("\nEvaluating Ensemble (RF + XGB)...")
evaluate_model(X_test, y_test, ensemble)

# Load DL Model
from tensorflow.keras.models import load_model
from scripts.train_dl_model import SimpleAttention

del X_test, y_test  # Clear memory for DL

# For DL evaluation
print("\n Evaluating Deep Learning (CNN + LSTM + Attention)...")
from scripts.train_dl_model import X_test as X_dl_test, y_test as y_dl_test

dl_model = load_model(os.path.join(MODEL_PATH, "best_model.h5"), custom_objects={'SimpleAttention': SimpleAttention})
evaluate_model(X_dl_test, y_dl_test, dl_model)