
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

TARGET = 'arrdel15'

def load_parquet(path):
    files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.parquet')]
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

def encode_strings(df):
    for col in df.select_dtypes(include=['object','string']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir',  default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--train',      default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train'))
    parser.add_argument('--validation', default=os.environ.get('SM_CHANNEL_VALIDATION', '/opt/ml/input/data/validation'))
    args = parser.parse_args()

    train_df = load_parquet(args.train)
    val_df   = load_parquet(args.validation)

    train_df = encode_strings(train_df)
    val_df   = encode_strings(val_df)

    X_train, y_train = train_df.drop(columns=[TARGET]).fillna(0), train_df[TARGET]
    X_val,   y_val   = val_df.drop(columns=[TARGET]).fillna(0),   val_df[TARGET]

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf',    LogisticRegression(max_iter=500, class_weight='balanced', random_state=42, n_jobs=-1))
    ])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:,1]

    metrics = {
        'accuracy':  round(float(accuracy_score(y_val,  y_pred)), 4),
        'precision': round(float(precision_score(y_val, y_pred, zero_division=0)), 4),
        'recall':    round(float(recall_score(y_val,    y_pred, zero_division=0)), 4),
        'f1':        round(float(f1_score(y_val,        y_pred, zero_division=0)), 4),
        'roc_auc':   round(float(roc_auc_score(y_val,   y_prob)), 4),
    }
    print('Validation metrics:', metrics)
    with open(os.path.join(args.model_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f)
    joblib.dump(model, os.path.join(args.model_dir, 'model.joblib'))
    print('Model saved.')

if __name__ == '__main__':
    main()
