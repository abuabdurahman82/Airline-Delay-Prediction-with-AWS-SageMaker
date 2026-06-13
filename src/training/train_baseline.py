import argparse, os, json
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

TARGET = "arrdel15"

def load_parquet(path):
    files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".parquet")]
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

def encode_strings(df):
    for col in df.select_dtypes(include=["object", "string"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df

def evaluate(model, X, y, split_name):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    metrics = {
        "split"     : split_name,
        "accuracy"  : round(float(accuracy_score(y, y_pred)), 4),
        "precision" : round(float(precision_score(y, y_pred, zero_division=0)), 4),
        "recall"    : round(float(recall_score(y, y_pred, zero_division=0)), 4),
        "f1"        : round(float(f1_score(y, y_pred, zero_division=0)), 4),
        "roc_auc"   : round(float(roc_auc_score(y, y_prob)), 4),
    }
    cm = confusion_matrix(y, y_pred).tolist()
    print(json.dumps({"metrics": metrics, "confusion_matrix": cm}))
    return metrics, cm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",           default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--validation",      default=os.environ.get("SM_CHANNEL_VALIDATION"))
    parser.add_argument("--model-dir",       default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--output-data-dir", default=os.environ.get("SM_OUTPUT_DATA_DIR"))
    parser.add_argument("--max-iter",        type=int,   default=300)
    parser.add_argument("--solver",          default="lbfgs")
    args = parser.parse_args()

    print("Loading data...")
    train_df = load_parquet(args.train)
    val_df   = load_parquet(args.validation)

    feature_cols = [c for c in train_df.columns if c != TARGET]
    X_train = encode_strings(train_df[feature_cols].fillna(0))
    y_train = train_df[TARGET]
    X_val   = encode_strings(val_df[feature_cols].fillna(0))
    y_val   = val_df[TARGET]

    print(f"Train : {X_train.shape} | delay rate {y_train.mean():.3f}")
    print(f"Val   : {X_val.shape}   | delay rate {y_val.mean():.3f}")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            max_iter=args.max_iter,
            C=1.0,
            solver=args.solver,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        ))
    ])

    print("Training Logistic Regression baseline...")
    model.fit(X_train, y_train)

    train_metrics, train_cm = evaluate(model, X_train, y_train, "train")
    val_metrics,   val_cm   = evaluate(model, X_val,   y_val,   "validation")

    os.makedirs(args.output_data_dir, exist_ok=True)
    output = {
        "model"      : "LogisticRegression",
        "train"      : {"metrics": train_metrics, "confusion_matrix": train_cm},
        "validation" : {"metrics": val_metrics,   "confusion_matrix": val_cm},
        "features"   : feature_cols,
        "hyperparameters": {"max_iter": args.max_iter, "solver": args.solver}
    }
    with open(os.path.join(args.output_data_dir, "baseline_metrics.json"), "w") as f:
        json.dump(output, f, indent=2)

    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))
    joblib.dump(feature_cols, os.path.join(args.model_dir, "feature_cols.joblib"))
    print("Model and feature list saved.")