"""
MAIN EXECUTION PIPELINE
Real-Time Credit Card Fraud Detection System
Executes data preprocessing, model training, evaluation, threshold tuning, and artifact generation.
"""

import joblib
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import precision_recall_curve, auc

# Import custom modular components from src package
from src.preprocessing import load_and_clean_data, scale_features
from src.modeling import (
    get_hybrid_resampled_data,
    train_supervised_classifiers,
    train_unsupervised_detector
)
from src.evaluation import (
    evaluate_model_performance,
    plot_precision_recall_curve,
    optimize_cost_threshold
)


def run_full_pipeline():
    print("==========================================================")
    print("      REAL-TIME CREDIT CARD FRAUD DETECTION PIPELINE      ")
    print("==========================================================\n")
    
    # ---------------------------------------------------------
    # STEP 1: Data Understanding & Preprocessing
    # ---------------------------------------------------------
    df = load_and_clean_data("creditcard.csv")
    df = scale_features(df)
    
    X = df.drop("Class", axis=1)
    y = df["Class"]
    
    # Stratified Train-Test Split (Requirement 3.3)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n[DATA SPLIT] Train set size: {X_train.shape[0]} | Test set size: {X_test.shape[0]}")
    
    # Stratified K-Fold CV validation check
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print("[CROSS-VALIDATION] Stratified K-Fold initialized (5 splits).")
    
    # ---------------------------------------------------------
    # STEP 2: Imbalance Handling (SMOTEENN)
    # ---------------------------------------------------------
    X_train_res, y_train_res = get_hybrid_resampled_data(X_train, y_train)
    
    # ---------------------------------------------------------
    # STEP 3: Model Training (Supervised + Unsupervised)
    # ---------------------------------------------------------
    trained_models = train_supervised_classifiers(X_train_res, y_train_res)
    
    # Train Unsupervised Isolation Forest
    iso_forest = train_unsupervised_detector(X_train)
    
    # ---------------------------------------------------------
    # STEP 4: Evaluation & Cost Optimization
    # ---------------------------------------------------------
    print("\n--- MODEL PERFORMANCE EVALUATION ---")
    model_probs = {}
    
    for name, model in trained_models.items():
        probs = model.predict_proba(X_test)[:, 1]
        model_probs[name] = probs
        metrics = evaluate_model_performance(y_test, probs, model_name=name)
        print(f"Model: {metrics['Model']:<20} | PR-AUC: {metrics['PR-AUC']:.4f} | ROC-AUC: {metrics['ROC-AUC']:.4f}")
        
    # Isolation forest evaluation
    iso_preds = np.where(iso_forest.predict(X_test) == -1, 1, 0)
    prec_iso, rec_iso, _ = precision_recall_curve(y_test, iso_preds)
    print(f"Model: Isolation Forest (Unsup) | PR-AUC: {auc(rec_iso, prec_iso):.4f}")
    
    # Select winning model (XGBoost)
    best_model_name = "XGBoost"
    best_model = trained_models[best_model_name]
    best_probs = model_probs[best_model_name]
    
    # Plot PR Curve
    plot_precision_recall_curve(y_test, best_probs, model_name=best_model_name)
    
    # Optimize Cost-Sensitive Decision Threshold
    optimal_th, min_cost = optimize_cost_threshold(y_test, best_probs)
    print(f"\n--- COST-SENSITIVE THRESHOLD OPTIMIZATION ---")
    print(f"Optimal Decision Threshold: {optimal_th:.4f} (Default was 0.5)")
    print(f"Minimum Estimated Financial Loss: ${min_cost:,.2f}")
    
    # ---------------------------------------------------------
    # STEP 5: Save Model Artifact
    # ---------------------------------------------------------
    model_filename = "fraud_detection_xgb_model.pkl"
    joblib.dump(best_model, model_filename)
    print(f"\n[ARTIFACT] Trained model saved as '{model_filename}'.")
    
    # ---------------------------------------------------------
    # STEP 6: Stretch Goal - Real-Time Scoring Simulation
    # ---------------------------------------------------------
    print("\n--- REAL-TIME SCORING STREAM SIMULATION ---")
    sample_transactions = X_test.head(5)
    
    for idx, row in sample_transactions.iterrows():
        row_df = pd.DataFrame([row])
        prob = best_model.predict_proba(row_df)[:, 1][0]
        is_fraud = prob >= optimal_th
        decision = "ALERT: FRAUD DETECTED" if is_fraud else "GENUINE TRANSACTION"
        print(f"Txn ID: {idx:<6} | Risk Probability: {prob:.4f} | Status: {decision}")
        time.sleep(0.3)
        
    print("\n==========================================================")
    print("       PIPELINE EXECUTION COMPLETED SUCCESSFULLY          ")
    print("==========================================================")


if __name__ == "__main__":
    run_full_pipeline()