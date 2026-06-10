# Runs inside SageMaker container
import argparse, os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

LOAD_COLS = [
  "flightdate", "year", "quarter", "month", "dayofmonth", "dayofweek",
  "reporting_airline", "origin", "dest", "originstate", "deststate",
  "crsdeptime", "crsarrtime", "crselapsedtime", "distance",
  "deptimeblk", "arrtimeblk", "arrdel15", "cancelled", "diverted"
]

LEAKAGE_COLS = [
  "depdelay", "depdelayminutes", "dep_del15",
  "arrdelay", "arrdelayminutes", "arrivaldelaygroups",
  "actualelapsedtime", "airtime", "taxiout", "taxiin",
  "wheelsoff", "wheelson", "carrierdelay", "weatherdelay",
  "nasdelay", "securitydelay", "lateaircraftdelay"
]

DROP_COLS = [
  "cancelled", "diverted", "quarter", "distancegroup",
  "crsdeptime", "crsarrtime", "deptimeblk", "arrtimeblk", "flightdate"
]

def load_raw_data(input_dir):
  dfs = []
  for root, dirs, files in os.walk(input_dir):
      for f in files:
          if f.endswith(".csv"):
              path = os.path.join(root, f)
              print(f"Loading {path}...")
              df = pd.read_csv(path, low_memory=False)
              df.columns = df.columns.str.lower()
              dfs.append(df)
  df = pd.concat(dfs, ignore_index=True)
  print(f"Raw shape: {df.shape}")
  return df

def clean(df):
  cols = [c for c in LOAD_COLS if c in df.columns]
  df = df[cols].copy()
  df = df[(df["cancelled"] == 0) & (df["diverted"] == 0)]
  df = df[df["arrdel15"].notna()]
  df["arrdel15"] = df["arrdel15"].astype(int)
  df["flightdate"] = pd.to_datetime(df["flightdate"])
  df = df.sort_values("flightdate").reset_index(drop=True)
  print(f"After cleaning: {df.shape}")
  print(f"Delay rate: {df['arrdel15'].mean():.1%}")
  return df

def engineer_features(df):
  df["dep_hour"]   = (df["crsdeptime"] // 100).astype(int).clip(0, 23)
  df["is_weekend"] = df["dayofweek"].isin([6, 7]).astype(int)
  df["route"]      = df["origin"] + "_" + df["dest"]
  return df

def time_based_split(df):
  n = len(df)
  train_end = int(n * 0.40)
  val_end   = int(n * 0.50)
  test_end  = int(n * 0.60)
  train = df.iloc[:train_end].copy()
  val   = df.iloc[train_end:val_end].copy()
  test  = df.iloc[val_end:test_end].copy()
  prod  = df.iloc[test_end:].copy()
  for name, split in [("train",train),("val",val),("test",test),("prod",prod)]:
      print(f"{name}: {split.shape} | {split['flightdate'].min()} to {split['flightdate'].max()}")
  return train, val, test, prod

def compute_historical_rates(train_df, target_dfs):
  global_rate = train_df["arrdel15"].mean()
  rate_maps = {
      "carrier_delay_rate": train_df.groupby("reporting_airline")["arrdel15"].mean(),
      "origin_delay_rate":  train_df.groupby("origin")["arrdel15"].mean(),
      "dest_delay_rate":    train_df.groupby("dest")["arrdel15"].mean(),
      "route_delay_rate":   train_df.groupby("route")["arrdel15"].mean(),
      "hour_delay_rate":    train_df.groupby("dep_hour")["arrdel15"].mean(),
  }
  results = []
  for df in target_dfs:
      df = df.copy()
      df["carrier_delay_rate"] = df["reporting_airline"].map(rate_maps["carrier_delay_rate"]).fillna(global_rate)
      df["origin_delay_rate"]  = df["origin"].map(rate_maps["origin_delay_rate"]).fillna(global_rate)
      df["dest_delay_rate"]    = df["dest"].map(rate_maps["dest_delay_rate"]).fillna(global_rate)
      df["route_delay_rate"]   = df["route"].map(rate_maps["route_delay_rate"]).fillna(global_rate)
      df["hour_delay_rate"]    = df["dep_hour"].map(rate_maps["hour_delay_rate"]).fillna(global_rate)
      results.append(df)
  return results

def encode_categoricals(train_df, target_dfs):
  label_cols = ["reporting_airline", "originstate", "deststate"]
  encoders = {}
  for col in label_cols:
      le = LabelEncoder()
      le.fit(train_df[col].astype(str))
      encoders[col] = le
  results = []
  for df in target_dfs:
      df = df.copy()
      for col, le in encoders.items():
          known = set(le.classes_)
          df[col] = df[col].astype(str).apply(
              lambda x: x if x in known else le.classes_[0]
          )
          df[col] = le.transform(df[col])
      results.append(df)
  return results, encoders

def drop_unneeded_columns(df):
  to_drop = [c for c in DROP_COLS + LEAKAGE_COLS if c in df.columns]
  return df.drop(columns=to_drop)

def save_splits(train, val, test, prod, output_dir):
  os.makedirs(f"{output_dir}/train", exist_ok=True)
  os.makedirs(f"{output_dir}/validation", exist_ok=True)
  os.makedirs(f"{output_dir}/test", exist_ok=True)
  os.makedirs(f"{output_dir}/production_simulation", exist_ok=True)
  train.to_parquet(f"{output_dir}/train/data.parquet", index=False)
  val.to_parquet(f"{output_dir}/validation/data.parquet", index=False)
  test.to_parquet(f"{output_dir}/test/data.parquet", index=False)
  prod.to_parquet(f"{output_dir}/production_simulation/data.parquet", index=False)
  print(f"train: {train.shape}")
  print(f"validation: {val.shape}")
  print(f"test: {test.shape}")
  print(f"production: {prod.shape}")
  print(f"features: {[c for c in train.columns if c != 'arrdel15']}")

def main(input_dir, output_dir):
  df = load_raw_data(input_dir)
  df = clean(df)
  df = engineer_features(df)
  train, val, test, prod = time_based_split(df)
  train, val, test, prod = compute_historical_rates(train, [train, val, test, prod])
  [train, val, test, prod], _ = encode_categoricals(train, [train, val, test, prod])
  train = drop_unneeded_columns(train)
  val   = drop_unneeded_columns(val)
  test  = drop_unneeded_columns(test)
  prod  = drop_unneeded_columns(prod)
  save_splits(train, val, test, prod, output_dir)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--input-dir",  default="/opt/ml/processing/input")
  parser.add_argument("--output-dir", default="/opt/ml/processing/output")
  args = parser.parse_args()
  main(args.input_dir, args.output_dir)