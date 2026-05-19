# Airline Delay Prediction with AWS SageMaker

## Project Overview

This repository contains an end-to-end machine learning operations implementation for predicting whether a U.S. domestic flight will arrive **15 minutes or more late**. The project is implemented primarily in **AWS SageMaker** and is designed to demonstrate a production-oriented MLOps workflow, including data ingestion, preprocessing, feature engineering, model training, evaluation, model registration, batch inference, monitoring, and CI/CD.

The project uses the **Bureau of Transportation Statistics Airline On-Time Performance / TranStats** dataset as the primary data source. The recommended modeling approach is a **single global binary classification model** trained across airlines, airports, and routes. XGBoost is used as the main model candidate.

The implementation of record for the course project should be the SageMaker-based workflow in this repository.

## Business Problem

Unexpected flight delays create missed connections, passenger dissatisfaction, crew scheduling disruptions, airport congestion, and increased operating costs. Airlines and airports benefit from an early warning signal that identifies flights at higher risk of arrival delay. This project creates a machine learning system that predicts whether a scheduled flight is likely to arrive at least 15 minutes late, allowing operations teams to prioritize high-risk flights and improve communication with passengers and internal stakeholders.

## Machine Learning Objective

The machine learning task is **supervised binary classification**. The target label is `arr_delay_15`.

| Label | Definition |
|---:|---|
| `0` | Flight arrived less than 15 minutes late or on time |
| `1` | Flight arrived 15 minutes or more late |

The project emphasizes **recall** as the primary operational metric because missing a high-risk delayed flight can be costly. The model also reports F1-score, precision, ROC-AUC, and confusion matrix results to provide a balanced evaluation.

## Dataset

The primary dataset is the official BTS Airline On-Time Performance dataset. BTS provides monthly flight-level records containing scheduled and actual departure and arrival times, airline, origin airport, destination airport, distance, cancellation, diversion, and delay fields. BTS reports that Airline On-Time Statistics are available from January 1995 through February 2026, which provides more than enough data for the course project requirements.[1]

The recommended working dataset is one full year of monthly files, such as January through December 2024. Monthly files should be stored separately in S3 to preserve raw-data lineage and satisfy the requirement for multiple raw data files.

## Proposed S3 Structure

```text
s3://<bucket-name>/airline-delay/
├── raw/
│   └── year=2024/month=01/
├── processed/
├── features/
├── training/
├── validation/
├── test/
├── production_simulation/
├── model_artifacts/
├── batch_predictions/
└── monitoring/
```

## Repository Structure

```text
airline-delay-prediction-sagemaker/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training_review.ipynb
│   ├── 04_inference_validation.ipynb
│   └── 05_monitoring_review.ipynb
├── src/
│   ├── preprocessing/
│   │   └── preprocess.py
│   ├── features/
│   │   └── build_features.py
│   ├── training/
│   │   ├── train_logistic_regression.py
│   │   └── train_xgboost.py
│   ├── evaluation/
│   │   └── evaluate_model.py
│   └── inference/
│       └── batch_inference.py
├── pipelines/
│   └── sagemaker_pipeline.py
├── configs/
│   ├── feature_config.yaml
│   ├── training_config.yaml
│   └── evaluation_thresholds.yaml
├── docs/
│   ├── architecture_diagram.png
│   ├── design_document.md
│   └── video_outline.md
├── reports/
│   ├── evaluation_report.md
│   ├── model_monitor_report.md
│   └── subgroup_analysis.md
└── .github/
    └── workflows/
        └── sagemaker-ci.yml
```

## SageMaker Architecture

The project should follow a SageMaker-first architecture. Raw data is stored in S3, then explored in SageMaker notebooks. Preprocessing logic is converted into repeatable SageMaker Processing jobs. Engineered features are stored in SageMaker Feature Store. Logistic Regression and XGBoost models are trained using SageMaker Training Jobs. The best model is registered in SageMaker Model Registry, then validated using SageMaker Batch Transform on simulated production data. Monitoring is demonstrated through SageMaker Model Monitor and CloudWatch. CI/CD is implemented through SageMaker Pipelines or GitHub Actions that trigger SageMaker jobs.

| Stage | AWS Component | Repository Location |
|---|---|---|
| Data storage | Amazon S3 | Documented in `README.md` and `configs/` |
| Exploration | SageMaker Studio / notebooks | `notebooks/01_eda.ipynb` |
| Processing | SageMaker Processing | `src/preprocessing/preprocess.py` |
| Feature engineering | SageMaker Feature Store | `src/features/build_features.py` |
| Training | SageMaker Training Jobs | `src/training/` |
| Evaluation | SageMaker Processing or notebook evaluation | `src/evaluation/evaluate_model.py` |
| Registry | SageMaker Model Registry | `pipelines/sagemaker_pipeline.py` |
| Inference | SageMaker Batch Transform | `src/inference/batch_inference.py` |
| Monitoring | SageMaker Model Monitor and CloudWatch | `reports/model_monitor_report.md` |
| CI/CD | SageMaker Pipelines or GitHub Actions | `pipelines/` or `.github/workflows/` |

## Feature Engineering

The initial global model should use flight schedule, route, carrier, airport, distance, and historical delay features. The team should carefully avoid data leakage by excluding any fields that would not be known before prediction time.

| Feature Category | Example Features |
|---|---|
| Carrier | `op_unique_carrier`, encoded carrier identifier |
| Airport | `origin`, `dest`, origin and destination airport encodings |
| Route | `origin_dest_route`, historical route delay rate |
| Time | month, day of week, scheduled departure hour |
| Distance | flight distance, distance bucket |
| Historical behavior | historical carrier delay rate, historical airport delay rate, historical route delay rate |

Leakage-prone fields such as actual arrival time, actual elapsed time, arrival delay, delay cause minutes, and other post-flight variables should be excluded from the training features.

## Model Training and Evaluation

The project trains two model families. Logistic Regression serves as a baseline because it is simple, interpretable, and useful for benchmarking. XGBoost is the main model because it is well suited for structured tabular data and is supported by SageMaker training workflows.

| Model | Purpose | Expected Use |
|---|---|---|
| XGBoost | Main model | Capture nonlinear interactions and improve predictive performance |

Evaluation should include recall, F1-score, precision, ROC-AUC, and a confusion matrix. Recall should receive special attention because the business cost of missing a risky delayed flight may be higher than the cost of flagging some flights that ultimately arrive on time.

## Deployment Strategy

The primary deployment method is **SageMaker Batch Transform**. Batch inference is appropriate because airlines and airports often plan around scheduled flights in advance. The team may optionally deploy a small SageMaker real-time endpoint for a short interactive demonstration, but this should not replace the batch inference workflow.

## Monitoring and CI/CD

Monitoring should include data quality checks, drift detection, model-quality tracking, and infrastructure visibility. The project should generate SageMaker Model Monitor reports and show relevant CloudWatch dashboard evidence. CI/CD should be implemented using either SageMaker Pipelines or GitHub Actions. The pipeline should demonstrate a successful run and a controlled failure or blocked deployment scenario.

## Setup Instructions

Create and activate the project environment in SageMaker Studio or a SageMaker notebook environment.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Configure AWS access through the SageMaker execution role rather than storing credentials in the repository. No AWS keys, secrets, or private configuration files should be committed.

## Suggested Execution Order

| Step | Action | Main Output |
|---:|---|---|
| 1 | Upload monthly BTS files to S3 | Raw data lake |
| 2 | Run EDA notebook | Data understanding and charts |
| 3 | Run preprocessing job | Cleaned and split datasets |
| 4 | Create and ingest Feature Store feature groups | Managed feature store |
| 5 | Train baseline and XGBoost models | Model artifacts |
| 6 | Evaluate models and select candidate | Evaluation report |
| 7 | Register selected model | Model Registry entry |
| 8 | Run Batch Transform | Prediction output |
| 9 | Configure monitoring | Monitoring reports and dashboard |
| 10 | Run CI/CD workflow | Pipeline success and failure evidence |

## Final Deliverables

The repository supports the following final project deliverables.

| Deliverable | Location or Evidence |
|---|---|
| ML System Design Document | `docs/design_document.md` |
| Codebase | Full GitHub repository |
| Video outline or transcript | `docs/video_outline.md` |
| Model evaluation report | `reports/evaluation_report.md` |
| Monitoring report | `reports/model_monitor_report.md` |
| Batch inference output evidence | S3 path documented in `reports/` |

## References

[1]: https://www.transtats.bts.gov/ontime/ — Bureau of Transportation Statistics, Airline On-Time Statistics  
[2]: https://aws.amazon.com/sagemaker/ — Amazon SageMaker overview
