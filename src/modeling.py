"""
Machine Learning Modeling Module
Implements Imbalance Handling (SMOTE, ADASYN, SMOTEENN), Supervised Classifiers,
and Unsupervised Anomaly Detection.
"""

import numpy as np
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from xgboost import XGBClassifier


def compare_resampling_strategies(X_train, y_train):
    """Compares different imbalance techniques using a baseline Logistic Regression model."""
    samplers = {
        "None (Original)": None,
        "Random Undersampling": RandomUnderSampler(random_state=42),
        "SMOTE": SMOTE(random_state=42),
        "ADASYN": ADASYN(random_state=42),
        "SMOTEENN (Hybrid)": SMOTEENN(random_state=42)
    }
    return samplers


def get_hybrid_resampled_data(X_train, y_train, random_state=42):
    """Applies SMOTEENN hybrid approach (combination of SMOTE oversampling and ENN undersampling).
    SMOTE synthesizes minority class instances, and ENN removes noisy overlap samples near boundary lines.
    """
    print("[MODELING] Resampling dataset using hybrid SMOTEENN strategy...")
    smote_enn = SMOTEENN(random_state=random_state)
    X_res, y_res = smote_enn.fit_resample(X_train, y_train)
    print(f"[MODELING] Resampled dataset shape: {X_res.shape} (Balanced Class Distribution)")
    return X_res, y_res


def train_supervised_classifiers(X_train, y_train):
    """Trains production supervised classifiers (Logistic Regression, Random Forest, XGBoost)."""
    print("[MODELING] Training Supervised Models...")
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=100, eval_metric="logloss", random_state=42)
    }
    
    trained_models = {}
    for name, model in models.items():
        print(f"[MODELING] Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        
    return trained_models


def train_unsupervised_detector(X_train, contamination=0.0017):
    """Trains an Isolation Forest anomaly detector for unsupervised fraud detection."""
    print("[MODELING] Training Isolation Forest Unsupervised Anomaly Detector...")
    iso_forest = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    iso_forest.fit(X_train)
    return iso_forest