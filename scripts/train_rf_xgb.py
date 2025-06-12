import time
import joblib
import json
import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from skopt import BayesSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score

from preprocessing import load_data, preprocess_group, encode_labels, scale_features

# Paths
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SAVE_PATH = os.path.join(ROOT, "save_point")
MODEL_PATH = os.path.join(ROOT, "models")

# Load and preprocess dataset
data = load_data()
data = data.groupby('name', group_keys=False).apply(preprocess_group)
data, label_encoder = encode_labels(data)

feature_cols = ['wavenumbers', 'transmittance']
data, scaler = scale_features(data, feature_cols, method='standard')

# Add scaled columns
data['scaled_wavenumbers'] = scaler.transform(data[['wavenumbers']])[:, 0]
data['scaled_transmittance'] = scaler.transform(data[['transmittance']])[:, 0]

# Feature matrix and labels
X = data[['scaled_wavenumbers', 'scaled_transmittance', 'first_derivative', 'second_derivative']]
y = data['name_encoded']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)

# Save datasets
np.save(os.path.join(SAVE_PATH, "X_train.npy"), X_train)
np.save(os.path.join(SAVE_PATH, "X_test.npy"), X_test)
np.save(os.path.join(SAVE_PATH, "y_train.npy"), y_train)
np.save(os.path.join(SAVE_PATH, "y_test.npy"), y_test)

# RF Tuning
rf = RandomForestClassifier(random_state=42, n_jobs=-1)
rf_param_grid = {
    "n_estimators": (15, 300),
    "max_depth": (5, 50),
    "min_samples_split": (2, 7),
    "min_samples_leaf": (1, 7),
    "class_weight": ["balanced", None]
}
bayes_rf = BayesSearchCV(rf, rf_param_grid, n_iter=15, cv=3, scoring='accuracy', random_state=42)
start_rf = time.time()
bayes_rf.fit(X_train, y_train)
best_rf = bayes_rf.best_estimator_
print(" RF Training Done in %.2fs" % (time.time() - start_rf))
print("Best RF Accuracy (CV):", bayes_rf.best_score_)
joblib.dump(best_rf, os.path.join(MODEL_PATH, "best_rf.pkl"))
with open(os.path.join(SAVE_PATH, "best_rf_params.json"), "w") as f:
    json.dump(bayes_rf.best_params_, f)

# XGB Tuning
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb_param_grid = {
    'n_estimators': (30, 300),
    'max_depth': (3, 20),
    'learning_rate': (0.05, 0.3, 'log-uniform'),
    'subsample': (0.6, 1.0, 'uniform'),
    'colsample_bytree': (0.6, 1.0, 'uniform')
}
bayes_xgb = BayesSearchCV(xgb, xgb_param_grid, n_iter=15, cv=3, scoring='accuracy', random_state=42)
start_xgb = time.time()
bayes_xgb.fit(X_train, y_train)
best_xgb = bayes_xgb.best_estimator_
print(" XGB Training Done in %.2fs" % (time.time() - start_xgb))
print("Best XGB Accuracy (CV):", bayes_xgb.best_score_)
joblib.dump(best_xgb, os.path.join(MODEL_PATH, "best_xgb.pkl"))
with open(os.path.join(SAVE_PATH, "best_xgb_params.json"), "w") as f:
    json.dump(bayes_xgb.best_params_, f)

# Ensemble Voting 
ensemble = VotingClassifier(
    estimators=[('rf', best_rf), ('xgb', best_xgb)],
    voting='soft'
)
ensemble.fit(X_train, y_train)
joblib.dump(ensemble, os.path.join(MODEL_PATH, "rf_xgb_ensemble.pkl"))

# Evaluation 
y_pred = ensemble.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cv_scores = cross_val_score(ensemble, X_train, y_train, cv=5)
print("\n Test Accuracy: %.2f%%" % (acc * 100))
print(" CV Accuracy: %.4f ± %.4f" % (cv_scores.mean(), cv_scores.std()))
print("\n Classification Report:\n", classification_report(y_test, y_pred))
