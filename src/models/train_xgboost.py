import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "master_pd_dataset.parquet"
BASELINE_MODEL_PATH = (
    PROJECT_ROOT / "models" / "baseline_logistic_model.joblib"
)
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_pd_model.joblib"
REPORT_PATH = PROJECT_ROOT / "outputs" / "reports" / "xgboost_report.txt"

TARGET_COLUMN = "TARGET"
ID_COLUMN = "SK_ID_CURR"
TEST_SIZE = 0.20
RANDOM_STATE = 42


def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Master dataset not found: {path}. "
            "Run src/features/build_master_dataset.py first."
        )

    LOGGER.info("Loading master dataset from %s", path)
    dataset = pd.read_parquet(path)

    if TARGET_COLUMN not in dataset.columns:
        raise KeyError(f"Dataset does not contain target column {TARGET_COLUMN}.")
    if dataset[TARGET_COLUMN].isna().any():
        missing_count = int(dataset[TARGET_COLUMN].isna().sum())
        raise ValueError(
            f"{TARGET_COLUMN} contains {missing_count:,} missing values."
        )

    target_values = set(dataset[TARGET_COLUMN].unique())
    if target_values != {0, 1}:
        raise ValueError(
            f"{TARGET_COLUMN} must contain binary values 0 and 1; "
            f"found {sorted(target_values)}."
        )

    LOGGER.info(
        "Loaded dataset with %s rows and %s columns",
        f"{dataset.shape[0]:,}",
        f"{dataset.shape[1]:,}",
    )
    return dataset


def build_pipeline(features: pd.DataFrame) -> Pipeline:
    numeric_columns = features.select_dtypes(include="number").columns.tolist()
    categorical_columns = features.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()
    unsupported_columns = sorted(
        set(features.columns) - set(numeric_columns) - set(categorical_columns)
    )

    if unsupported_columns:
        raise TypeError(
            "Unsupported feature dtypes for columns: "
            + ", ".join(unsupported_columns)
        )
    if not numeric_columns and not categorical_columns:
        raise ValueError("No model features were found.")

    LOGGER.info(
        "Preparing %s numeric and %s categorical features",
        f"{len(numeric_columns):,}",
        f"{len(categorical_columns):,}",
    )

    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot_encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            ),
        ]
    )

    transformers = []
    if numeric_columns:
        transformers.append(("numeric", numeric_pipeline, numeric_columns))
    if categorical_columns:
        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        sparse_threshold=1.0,
    )
    classifier = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def calculate_metrics(model: Pipeline, features, target) -> dict:
    predicted_labels = model.predict(features)
    predicted_probabilities = model.predict_proba(features)[:, 1]
    matrix = confusion_matrix(target, predicted_labels, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()

    return {
        "roc_auc": roc_auc_score(target, predicted_probabilities),
        "precision": precision_score(
            target, predicted_labels, zero_division=0
        ),
        "recall": recall_score(target, predicted_labels, zero_division=0),
        "f1": f1_score(target, predicted_labels, zero_division=0),
        "true_negative": int(true_negative),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "true_positive": int(true_positive),
    }


def load_baseline_metrics(features, target) -> dict | None:
    if not BASELINE_MODEL_PATH.is_file():
        LOGGER.warning(
            "Baseline model not found at %s; comparison will be omitted.",
            BASELINE_MODEL_PATH,
        )
        return None

    LOGGER.info("Evaluating baseline logistic model on the same test set")
    baseline_model = joblib.load(BASELINE_MODEL_PATH)
    return calculate_metrics(baseline_model, features, target)


def create_report(
    y_train: pd.Series,
    y_test: pd.Series,
    xgboost_metrics: dict,
    baseline_metrics: dict | None,
) -> str:
    lines = [
        "XGBoost PD Model Report",
        "=======================",
        "",
        f"Dataset: {DATASET_PATH}",
        f"Training observations: {len(y_train):,}",
        f"Test observations: {len(y_test):,}",
        f"Test size: {TEST_SIZE:.0%}",
        f"Random state: {RANDOM_STATE}",
        "",
        "TARGET distribution",
        "-------------------",
        f"Train defaults (TARGET=1): {int(y_train.sum()):,} "
        f"({y_train.mean():.2%})",
        f"Test defaults (TARGET=1): {int(y_test.sum()):,} "
        f"({y_test.mean():.2%})",
        "",
        "XGBoost evaluation metrics",
        "--------------------------",
        f"ROC-AUC:   {xgboost_metrics['roc_auc']:.6f}",
        f"Precision: {xgboost_metrics['precision']:.6f}",
        f"Recall:    {xgboost_metrics['recall']:.6f}",
        f"F1:        {xgboost_metrics['f1']:.6f}",
        "",
        "Confusion matrix",
        "----------------",
        "                 Predicted 0  Predicted 1",
        f"Actual 0       {xgboost_metrics['true_negative']:11,d}  "
        f"{xgboost_metrics['false_positive']:11,d}",
        f"Actual 1       {xgboost_metrics['false_negative']:11,d}  "
        f"{xgboost_metrics['true_positive']:11,d}",
    ]

    if baseline_metrics is not None:
        lines.extend(
            [
                "",
                "Comparison with baseline logistic regression",
                "--------------------------------------------",
                "Metric       Logistic    XGBoost     Difference",
                f"ROC-AUC      {baseline_metrics['roc_auc']:.6f}    "
                f"{xgboost_metrics['roc_auc']:.6f}    "
                f"{xgboost_metrics['roc_auc'] - baseline_metrics['roc_auc']:+.6f}",
                f"Precision    {baseline_metrics['precision']:.6f}    "
                f"{xgboost_metrics['precision']:.6f}    "
                f"{xgboost_metrics['precision'] - baseline_metrics['precision']:+.6f}",
                f"Recall       {baseline_metrics['recall']:.6f}    "
                f"{xgboost_metrics['recall']:.6f}    "
                f"{xgboost_metrics['recall'] - baseline_metrics['recall']:+.6f}",
                f"F1           {baseline_metrics['f1']:.6f}    "
                f"{xgboost_metrics['f1']:.6f}    "
                f"{xgboost_metrics['f1'] - baseline_metrics['f1']:+.6f}",
            ]
        )

    lines.append("")
    return "\n".join(lines)


def train_xgboost() -> Pipeline:
    dataset = load_dataset()
    excluded_columns = [TARGET_COLUMN]
    if ID_COLUMN in dataset.columns:
        excluded_columns.append(ID_COLUMN)
        LOGGER.info("Excluding identifier column %s", ID_COLUMN)

    features = dataset.drop(columns=excluded_columns)
    target = dataset[TARGET_COLUMN].astype("int8")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    pipeline = build_pipeline(features)
    LOGGER.info("Training XGBoost classifier")
    pipeline.fit(x_train, y_train)

    xgboost_metrics = calculate_metrics(pipeline, x_test, y_test)
    baseline_metrics = load_baseline_metrics(x_test, y_test)
    report = create_report(
        y_train,
        y_test,
        xgboost_metrics,
        baseline_metrics,
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    REPORT_PATH.write_text(report, encoding="utf-8")

    LOGGER.info("Saved trained pipeline to %s", MODEL_PATH)
    LOGGER.info("Saved evaluation report to %s", REPORT_PATH)
    print(report)
    return pipeline


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    try:
        train_xgboost()
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        LOGGER.error("%s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
