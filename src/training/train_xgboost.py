import argparse, os, json
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
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

def evaluate(model, dmatrix, y, split_name):
    y_prob = model.predict(dmatrix)
    y_pred = (y_prob >= 0.5).astype(int)
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
    parser.add_argument("--train",            default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--validation",       default=os.environ.get("SM_CHANNEL_VALIDATION"))
    parser.add_argument("--model_dir",        default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--output_data_dir",  default=os.environ.get("SM_OUTPUT_DATA_DIR"))
    parser.add_argument("--num_round",        type=int,   default=200)
    parser.add_argument("--max_depth",        type=int,   default=6)
    parser.add_argument("--eta",              type=float, default=0.1)
    parser.add_argument("--subsample",        type=float, default=0.8)
    parser.add_argument("--colsample_bytree", type=float, default=0.8)
    parser.add_argument("--min_child_weight", type=int,   default=5)
    parser.add_argument("--scale_pos_weight", type=float, default=4.0)
    args = parser.parse_args()

    print("Loading data...")
    train_df = load_parquet(args.train)
    val_df   = load_parquet(args.validation)

    feature_cols = [c for c in train_df.columns if c != TARGET]
    X_train = encode_strings(train_df[feature_cols].fillna(0)).values
    y_train = train_df[TARGET].values
    X_val   = encode_strings(val_df[feature_cols].fillna(0)).values
    y_val   = val_df[TARGET].values

    print(f"Train : {X_train.shape} | delay rate {y_train.mean():.3f}")
    print(f"Val   : {X_val.shape}   | delay rate {y_val.mean():.3f}")

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
    dval   = xgb.DMatrix(X_val,   label=y_val,   feature_names=feature_cols)

    params = {
        "objective"        : "binary:logistic",
        "eval_metric"      : ["auc", "logloss"],
        "max_depth"        : args.max_depth,
        "eta"              : args.eta,
        "subsample"        : args.subsample,
        "colsample_bytree" : args.colsample_bytree,
        "min_child_weight" : args.min_child_weight,
        "scale_pos_weight" : args.scale_pos_weight,
        "seed"             : 42,
    }

    print("Training XGBoost...")
    evals_result = {}
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=args.num_round,
        evals=[(dtrain, "train"), (dval, "validation")],
        early_stopping_rounds=20,
        evals_result=evals_result,
        verbose_eval=50
    )

    best_iter = model.best_iteration
    val_auc   = evals_result["validation"]["auc"][best_iter]
    print(f"validation:auc={val_auc}")
    print(f"Best iteration: {best_iter}")

    train_metrics, train_cm = evaluate(model, dtrain, y_train, "train")
    val_metrics,   val_cm   = evaluate(model, dval,   y_val,   "validation")

    importance = model.get_score(importance_type="gain")
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15]
    print("Top 15 features by gain:")
    for feat, score in top_features:
        print(f"  {feat}: {score:.2f}")

    os.makedirs(args.output_data_dir, exist_ok=True)
    output = {
        "model"           : "XGBoost",
        "best_iteration"  : best_iter,
        "train"           : {"metrics": train_metrics, "confusion_matrix": train_cm},
        "validation"      : {"metrics": val_metrics,   "confusion_matrix": val_cm},
        "top_features"    : dict(top_features),
        "hyperparameters" : vars(args),
    }
    with open(os.path.join(args.output_data_dir, "xgb_metrics.json"), "w") as fp:
        json.dump(output, fp, indent=2)

    model.save_model(os.path.join(args.model_dir, "xgboost-model"))
    joblib.dump(feature_cols, os.path.join(args.model_dir, "feature_cols.joblib"))
    print("Model and feature list saved.")