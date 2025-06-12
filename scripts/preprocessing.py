import numpy as np
import pandas as pd
import joblib
from scipy.signal import savgol_filter
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import os

# Set paths
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SAVE_PATH = os.path.join(ROOT, "save_point")
DATA_PATH = os.path.join(ROOT, "data")
MODEL_PATH = os.path.join(ROOT, "models")

# Ensure directories exist
os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

def preprocess_group(group):
    group = group.sort_values('wavenumbers').reset_index(drop=True)
    T = group['transmittance'] / group['transmittance'].max()
    A = -np.log10(np.clip(T, 1e-8, 1))
    A_smooth = savgol_filter(A, window_length=11, polyorder=3)
    dA = np.gradient(A_smooth)
    ddA = np.gradient(dA)

    group['transmittance'] = T
    group['absorbance'] = A
    group['first_derivative'] = dA
    group['second_derivative'] = ddA
    return group

def compute_features(df):
    df = df.sort_values(by='wavenumbers').reset_index(drop=True)

    epsilon = 1e-8
    grad_wavenumbers = np.gradient(df['wavenumbers'])
    grad_transmittance = np.gradient(df['transmittance'])

    df['gradient_transmittance'] = grad_transmittance / (grad_wavenumbers + epsilon)
    df['curvature_transmittance'] = np.gradient(df['gradient_transmittance']) / (grad_wavenumbers + epsilon)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df

def encode_labels(df, label_column='name'):
    label_encoder = LabelEncoder()
    df['name_encoded'] = label_encoder.fit_transform(df[label_column])
    joblib.dump(label_encoder, os.path.join(SAVE_PATH, "label_encoder.pkl"))
    return df, label_encoder

def scale_features(df, feature_cols, method='standard'):
    if method == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    joblib.dump(scaler, os.path.join(SAVE_PATH, "scaler.pkl"))
    return df, scaler

def load_data(filename="Final_Standard_Data.xlsx"):
    path = os.path.join(DATA_PATH, filename)
    df = pd.read_excel(path)
    df.dropna(subset=['wavenumbers', 'transmittance', 'name'], inplace=True)
    return df
