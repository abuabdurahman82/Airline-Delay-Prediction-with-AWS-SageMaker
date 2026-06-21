
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
    parser.add_argument('--model-dir',      default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train',          default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation',     default=os.environ.get('SM_CHANNEL_VALIDATION'))
    parser.add_argument('--max_depth',      type=int,   default=6)
    parser.add_argument('--eta',            type=float, default=0.1)
    parser.add_argument('--subsample',      type=float, default=0.8)
    parser.add_argument('--colsample_bytree', type=float, default=0.8)
    parser.add_argument('--num_round',      type=int,   default=200)
    parser.add_argument('--scale_pos_weight', type=float, default=4.0)
    args = parser.parse_args()

    train_df = encode_strings(load_parquet(args.train))
    val_df   = encode_strings(load_parquet(args.validation))

    X_train, y_train = train_df.drop(columns=[TARGET]).fillna(0), train_df[TARGET]
    X_val,   y_val   = val_df.drop(columns=[TARGET]).fillna(0),   val_df[TARGET]

    feature_cols = list(X_train.columns)
    joblib.dump(feature_cols, os.path.join(args.model_dir, 'feature_cols.joblib'))

    dtrain = xgb.DMatrix(X_train.values, label=y_train, feature_names=feature_cols)
    dval   = xgb.DMatrix(X_val.values,   label=y_val,   feature_names=feature_cols)

    params = {
        'objective':        'binary:logistic',
        'eval_metric':      'auc',
        'max_depth':        args.max_depth,
        'eta':              args.eta,
        'subsample':        args.subsample,
        'colsample_bytree': args.colsample_bytree,
        'scale_pos_weight': args.scale_pos_weight,
        'seed':             42,
    }

    model = xgb.train(
        params, dtrain,
        num_boost_round=args.num_round,
        evals=[(dval, 'validation')],
        early_stopping_rounds=20,
        verbose_eval=50,
    )
    model.save_model(os.path.join(args.model_dir, 'xgboost-model'))

    y_prob = model.predict(dval)
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {
        'validation:auc': round(float(roc_auc_score(y_val, y_prob)), 4),
        'recall':         round(float(recall_score(y_val, y_pred, zero_division=0)), 4),
        'f1':             round(float(f1_score(y_val, y_pred, zero_division=0)), 4),
    }
    print('Validation metrics:', metrics)
    with open(os.path.join(args.model_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f)
    print('XGBoost model saved.')

if __name__ == '__main__':
    main()
