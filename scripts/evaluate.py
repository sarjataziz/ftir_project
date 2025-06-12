import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import joblib

def evaluate_model(X_test, y_test, model, class_names=None):
    y_pred = model.predict(X_test)
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)
    else:
        y_prob = y_pred

    if y_test.ndim > 1:
        y_test_cls = np.argmax(y_test, axis=1)
        y_pred_cls = np.argmax(y_pred, axis=1)
    else:
        y_test_cls = y_test
        y_pred_cls = y_pred

    print("\n Classification Report:")
    print(classification_report(y_test_cls, y_pred_cls, target_names=class_names))

    f1 = f1_score(y_test_cls, y_pred_cls, average='weighted')
    print(f"Weighted F1 Score: {f1:.4f}")

    # Confusion Matrix
    plt.figure(figsize=(10, 6))
    cm = confusion_matrix(y_test_cls, y_pred_cls)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    # ROC for Class 1 only 
    if y_prob.shape[1] > 1:
        fpr, tpr, _ = roc_curve(label_binarize(y_test_cls, classes=np.unique(y_test_cls))[:, 1],
                                y_prob[:, 1])
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
