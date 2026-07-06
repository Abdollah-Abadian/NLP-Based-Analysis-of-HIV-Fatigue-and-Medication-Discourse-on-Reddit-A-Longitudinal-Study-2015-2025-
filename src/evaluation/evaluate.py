from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    accuracy_score, hamming_loss, roc_auc_score, average_precision_score
)
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


def evaluate_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray = None
) -> Dict[str, float]:
    """Compute multi-label metrics."""
    metrics = {
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'micro_f1': f1_score(y_true, y_pred, average='micro', zero_division=0),
        'exact_match': accuracy_score(y_true, y_pred),  # exact match ratio
        'hamming_loss': hamming_loss(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
    }
    if y_proba is not None:
        # AUC-ROC (macro)
        roc = roc_auc_score(y_true, y_proba, average='macro', multi_class='ovr')
        pr_auc = average_precision_score(y_true, y_proba, average='macro')
        metrics['macro_auc_roc'] = roc
        metrics['macro_pr_auc'] = pr_auc
    return metrics


def run_baselines(X_train, y_train, X_test, y_test, vectorizer):
    """Train and evaluate baseline models."""
    baselines = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Linear SVM': LinearSVC(max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100),
        'LightGBM': LGBMClassifier(n_estimators=100)
    }
    results = {}
    for name, clf in baselines.items():
        # Wrap for multi-output
        multi_clf = MultiOutputClassifier(clf)
        multi_clf.fit(vectorizer.transform(X_train), y_train)
        y_pred = multi_clf.predict(vectorizer.transform(X_test))
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        results[name] = f1
    return results


def run_ablation(
    model,
    test_loader,
    device,
    variants: Dict[str, Dict]
) -> Dict[str, float]:
    """
    Run ablation experiments by modifying model/config.
    variants: dict of variant_name -> parameters to change.
    Returns macro F1 per variant.
    """
    # Implementation would load different model variants
    # For brevity, placeholder structure
    pass
