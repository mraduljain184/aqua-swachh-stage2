"""
Aqua-Swachh Machine Learning Model Training Pipeline
====================================================
Implements ML best practices for Stage 3 prototype filter condition classification:
1. Data loading and schema validation
2. Strict featurization ordering (Train/Test split before scaler fitting)
3. Model baseline comparison:
   - Baseline 1: Dummy (Majority Class)
   - Baseline 2: Logistic Regression (linear baseline)
   - Primary: Random Forest Classifier (ensemble)
4. Cross-validation & comprehensive evaluation (F1, Precision, Recall, Confusion Matrix)
5. Model export with joblib

Two model variants are trained on the same train/test split:
- Full model     (TDS x2 + temperature + flow)  -> random_forest_filter_model.joblib
- No-flow model  (TDS x2 + temperature only)    -> random_forest_filter_model_no_flow.joblib

The no-flow model is used automatically when a reading has no flow-meter data
(the current hardware build has no YF-S201 fitted). It cannot see flow
choking, so its scores are expected to be lower; they are reported honestly.
"""

import os
import json
import joblib
import pandas as pd
from typing import Dict, Any

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

from model.features import extract_features, feature_columns

CLASSES = ['Normal', 'Monitor', 'Service Required']
PRIMARY_MODEL_NAME = "Random Forest Classifier (Primary Model)"
MODEL_FILENAMES = {
    True: "random_forest_filter_model.joblib",
    False: "random_forest_filter_model_no_flow.joblib",
}


def _candidate_models(random_state: int) -> Dict[str, Pipeline]:
    return {
        "Dummy (Majority Class Baseline)": Pipeline([
            ('scaler', StandardScaler()),
            ('clf', DummyClassifier(strategy='most_frequent'))
        ]),
        "Logistic Regression (Linear Baseline)": Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=random_state))
        ]),
        PRIMARY_MODEL_NAME: Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(
                n_estimators=40,
                max_depth=6,
                min_samples_split=4,
                class_weight='balanced',
                random_state=random_state,
                n_jobs=1
            ))
        ])
    }


def _train_variant(df_train, df_test, y_train, y_test, include_flow: bool, random_state: int) -> Dict[str, Any]:
    cols = feature_columns(include_flow)
    X_train = extract_features(df_train, include_flow=include_flow)
    X_test = extract_features(df_test, include_flow=include_flow)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    comparison_results = {}

    for name, pipeline in _candidate_models(random_state).items():
        print(f"Evaluating {name}...", flush=True)
        # Cross validation on Training set only
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1_macro', n_jobs=1)

        # Fit on entire Training set, evaluate on unseen Test set
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred, average='macro')

        comparison_results[name] = {
            'cv_f1_mean': float(cv_scores.mean()),
            'cv_f1_std': float(cv_scores.std()),
            'test_accuracy': float(test_acc),
            'test_f1_macro': float(test_f1),
            'pipeline': pipeline,
            'y_pred': y_pred
        }
        print(f"  -> 5-Fold CV Macro F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})", flush=True)
        print(f"  -> Test Accuracy:     {test_acc:.4f}", flush=True)
        print(f"  -> Test Macro F1:      {test_f1:.4f}", flush=True)

    best = comparison_results[PRIMARY_MODEL_NAME]
    print("\n" + "-" * 60, flush=True)
    print(f"Classification Report for {PRIMARY_MODEL_NAME}:", flush=True)
    print(classification_report(y_test, best['y_pred'], labels=CLASSES, target_names=CLASSES), flush=True)

    cm = confusion_matrix(y_test, best['y_pred'], labels=CLASSES)
    print("Confusion Matrix (Rows: Actual, Columns: Predicted):", flush=True)
    print(f"Labels: {CLASSES}", flush=True)
    print(cm, flush=True)

    importances = best['pipeline'].named_steps['clf'].feature_importances_
    feat_imp = sorted(zip(cols, importances), key=lambda x: x[1], reverse=True)
    print("\nFeature Importances (Random Forest):", flush=True)
    for feat, imp in feat_imp:
        print(f"  - {feat:22s}: {imp:.4f}", flush=True)

    summary = {
        'best_model': PRIMARY_MODEL_NAME,
        'uses_flow_rate': include_flow,
        'feature_columns': cols,
        'classes': CLASSES,
        'test_accuracy': best['test_accuracy'],
        'test_macro_f1': best['test_f1_macro'],
        'confusion_matrix': cm.tolist(),
        'feature_importances': {feat: float(imp) for feat, imp in feat_imp},
        'comparison': {
            k: {
                'cv_f1_mean': v['cv_f1_mean'],
                'cv_f1_std': v['cv_f1_std'],
                'test_accuracy': v['test_accuracy'],
                'test_f1_macro': v['test_f1_macro']
            } for k, v in comparison_results.items()
        }
    }
    return {'summary': summary, 'pipeline': best['pipeline']}


def train_and_evaluate(
    data_path: str = "data/sensor_data.csv",
    model_output_dir: str = "model/saved_models",
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    print("=" * 60, flush=True)
    print("Aqua-Swachh: Filter Health ML Training Pipeline", flush=True)
    print("=" * 60, flush=True)

    # 1. Load Data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please generate or provide sensor CSV first.")

    df = pd.read_csv(data_path)
    print(f"\n[1/5] Loaded dataset from {data_path} with {len(df)} rows.", flush=True)

    # 2. Data Cleaning & Validation (Per ML Best Practices)
    target_col = 'filter_status'
    required_cols = ['tds_in', 'tds_out', 'temperature', 'flow_rate', target_col]

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in CSV: {missing_cols}")

    null_counts = df[required_cols].isnull().sum()
    print("Null value counts per feature:", flush=True)
    for col, count in null_counts.items():
        print(f"  - {col}: {count}", flush=True)
    if null_counts.sum() > 0:
        print("Dropping rows with missing values...", flush=True)
        df = df.dropna(subset=required_cols).reset_index(drop=True)

    valid_mask = (df['tds_in'] > 0) & (df['tds_out'] >= 0) & (df['temperature'] > 0) & (df['flow_rate'] >= 0)
    invalid_rows = (~valid_mask).sum()
    if invalid_rows > 0:
        print(f"Filtering out {invalid_rows} physically invalid sensor rows.", flush=True)
        df = df[valid_mask].reset_index(drop=True)

    print(f"Class distribution in clean dataset:\n{df[target_col].value_counts()}", flush=True)

    # 3. Strict Train/Test Split BEFORE Scaling/Fitting (shared by both variants)
    print(f"\n[2/5] Splitting data: {int((1-test_size)*100)}% Train / {int(test_size*100)}% Test (Stratified)...", flush=True)
    y = df[target_col].values
    df_train, df_test, y_train, y_test = train_test_split(
        df, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Training samples: {len(df_train)} | Test samples: {len(df_test)}", flush=True)

    # 4. Full sensor model
    print("\n[3/5] Training FULL model (TDS + temperature + flow):", flush=True)
    full = _train_variant(df_train, df_test, y_train, y_test, include_flow=True, random_state=random_state)

    # 5. No-flow model (current hardware: no flow meter fitted)
    print("\n[4/5] Training NO-FLOW model (TDS + temperature only):", flush=True)
    no_flow = _train_variant(df_train, df_test, y_train, y_test, include_flow=False, random_state=random_state)

    # 6. Save Model Artifacts
    print("\n[5/5] Exporting models and artifacts...", flush=True)
    os.makedirs(model_output_dir, exist_ok=True)
    for include_flow, variant in ((True, full), (False, no_flow)):
        path = os.path.join(model_output_dir, MODEL_FILENAMES[include_flow])
        joblib.dump(variant['pipeline'], path)
        print(f"Saved trained pipeline to: {path}", flush=True)

    # Top-level keys describe the full model (backwards compatible);
    # the no-flow model is reported under 'no_flow_model'.
    metrics_summary = dict(full['summary'])
    metrics_summary['training_data'] = os.path.basename(data_path)
    metrics_summary['no_flow_model'] = no_flow['summary']

    metrics_path = os.path.join(model_output_dir, "metrics_summary.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"Saved evaluation metrics to: {metrics_path}", flush=True)

    return metrics_summary


if __name__ == "__main__":
    train_and_evaluate()
