import os
import numpy as np
import pandas as pd
import joblib

from preprocessing import preprocess_group, compute_features

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SAVE_PATH = os.path.join(ROOT, "save_point")
MODEL_PATH = os.path.join(ROOT, "models")

def load_input_csv(csv_path):
    df = pd.read_csv(csv_path)
    if 'transmittance' not in df.columns and 'absorbance' in df.columns:
        df['transmittance'] = 10 ** (-df['absorbance']) * 100
    elif 'transmittance' in df.columns:
        df['absorbance'] = -np.log10(np.clip(df['transmittance'] / 100, 1e-8, 1))
    else:
        raise ValueError("CSV must have either 'transmittance' or 'absorbance'")
    df = df.dropna(subset=['wavenumbers', 'transmittance'])
    return df

def predict_model(csv_path, model_type="ensemble"):
    df = load_input_csv(csv_path)
    df = df.sort_values('wavenumbers').reset_index(drop=True)
    df = preprocess_group(df)

    # Load scaler & encoder
    scaler = joblib.load(os.path.join(SAVE_PATH, "scaler.pkl"))
    encoder = joblib.load(os.path.join(SAVE_PATH, "label_encoder.pkl"))

    # Apply scaling
    df['scaled_wavenumbers'] = scaler.transform(df[['wavenumbers']])[:, 0]
    df['scaled_transmittance'] = scaler.transform(df[['transmittance']])[:, 0]

    # Prepare feature input
    X_input = df[['scaled_wavenumbers', 'scaled_transmittance', 'first_derivative', 'second_derivative']].values

    # Load model
    if model_type == "rf":
        model = joblib.load(os.path.join(MODEL_PATH, "best_rf.pkl"))
    elif model_type == "xgb":
        model = joblib.load(os.path.join(MODEL_PATH, "best_xgb.pkl"))
    elif model_type == "ensemble":
        model = joblib.load(os.path.join(MODEL_PATH, "rf_xgb_ensemble.pkl"))
    else:
        raise ValueError("Invalid model_type. Choose from: 'rf', 'xgb', 'ensemble'.")

    # Predict
    preds = model.predict(X_input)
    predicted_names = encoder.inverse_transform(preds)
    most_common = pd.Series(predicted_names).value_counts().idxmax()

    print(f"\n[{model_type.upper()}] Most Frequent Predicted Compound: **{most_common}**")
    return predicted_names


def predict_dl(csv_path, sequence_length=200):
    from tensorflow.keras.models import load_model
    from train_dl_model import compute_features, SimpleAttention
    from tensorflow.keras.utils import to_categorical

    df = load_input_csv(csv_path)
    df = compute_features(df)

    # Load encoder & scaler
    encoder = joblib.load(os.path.join(SAVE_PATH, "label_encoder.pkl"))
    scaler = joblib.load(os.path.join(SAVE_PATH, "scaler.pkl"))
    df[['wavenumbers', 'transmittance', 'gradient_transmittance', 'curvature_transmittance']] = scaler.transform(
        df[['wavenumbers', 'transmittance', 'gradient_transmittance', 'curvature_transmittance']])

    # Prepare sequence
    X = df[['wavenumbers', 'transmittance', 'gradient_transmittance', 'curvature_transmittance']].values
    X_seq = [X[i:i + sequence_length] for i in range(len(X) - sequence_length)]
    if not X_seq:
        print(" Not enough data to form a valid sequence for DL model.")
        return
    X_seq = np.array(X_seq)

    # Load model
    model = load_model(os.path.join(MODEL_PATH, "best_model.h5"), custom_objects={"SimpleAttention": SimpleAttention})
    y_pred = model.predict(X_seq)
    preds = np.argmax(y_pred, axis=1)
    names = encoder.inverse_transform(preds)

    most_common = pd.Series(names).value_counts().idxmax()
    print(f"\n DL Prediction – Most Frequent Compound: **{most_common}**")
    return names

if __name__ == "__main__":
    test_file = os.path.join(ROOT, "data", "12.csv") 

    print("\n [RF] Running Random Forest Prediction:")
    predict_model(test_file, model_type="rf")

    print("\n [XGB] Running XGBoost Prediction:")
    predict_model(test_file, model_type="xgb")

    print("\n [ENSEMBLE] Running Ensemble Prediction:")
    predict_model(test_file, model_type="ensemble")

    print("\n [DL] Running Deep Learning Prediction:")
    predict_dl(test_file)


