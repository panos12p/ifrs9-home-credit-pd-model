import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "master_pd_dataset.parquet"
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_pd_model.joblib"
REPORT_PATH = (
    PROJECT_ROOT / "outputs" / "reports" / "xgboost_threshold_analysis.csv"
)
FIGURE_PATH = (
    PROJECT_ROOT / "outputs" / "figures" / "xgboost_threshold_performance.png"
)

TARGET_COLUMN = "TARGET"
ID_COLUMN = "SK_ID_CURR"
TEST_SIZE = 0.20
RANDOM_STATE = 42
THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]


def load_test_data() -> tuple[pd.DataFrame, pd.Series]:
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Master dataset not found: {DATASET_PATH}")

    dataset = pd.read_parquet(DATASET_PATH)
    if TARGET_COLUMN not in dataset.columns:
        raise KeyError(f"Dataset does not contain {TARGET_COLUMN}.")
    if dataset[TARGET_COLUMN].isna().any():
        raise ValueError(f"{TARGET_COLUMN} contains missing values.")

    target_values = set(dataset[TARGET_COLUMN].unique())
    if target_values != {0, 1}:
        raise ValueError(
            f"{TARGET_COLUMN} must contain 0 and 1; found {sorted(target_values)}."
        )

    excluded_columns = [TARGET_COLUMN]
    if ID_COLUMN in dataset.columns:
        excluded_columns.append(ID_COLUMN)

    features = dataset.drop(columns=excluded_columns)
    target = dataset[TARGET_COLUMN].astype("int8")

    _, x_test, _, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    LOGGER.info(
        "Reconstructed test set with %s observations and %.2f%% defaults",
        f"{len(y_test):,}",
        y_test.mean() * 100,
    )
    return x_test, y_test


def evaluate_thresholds(
    y_true: pd.Series,
    predicted_probabilities,
) -> pd.DataFrame:
    results = []

    for threshold in THRESHOLDS:
        predicted_labels = (predicted_probabilities >= threshold).astype("int8")
        results.append(
            {
                "threshold": threshold,
                "precision": precision_score(
                    y_true, predicted_labels, zero_division=0
                ),
                "recall": recall_score(
                    y_true, predicted_labels, zero_division=0
                ),
                "f1": f1_score(y_true, predicted_labels, zero_division=0),
                "predicted_default_rate": predicted_labels.mean(),
            }
        )

    return pd.DataFrame(results)


def create_performance_plot(results: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(10, 6))
    metric_styles = {
        "precision": ("Precision", "#287A5B", "o"),
        "recall": ("Recall", "#B33A3A", "s"),
        "f1": ("F1", "#3567A8", "^"),
        "predicted_default_rate": ("Predicted default rate", "#8A5A9E", "D"),
    }

    for column, (label, color, marker) in metric_styles.items():
        axis.plot(
            results["threshold"],
            results[column],
            label=label,
            color=color,
            marker=marker,
            linewidth=2,
        )

    best_row = results.loc[results["f1"].idxmax()]
    axis.axvline(
        best_row["threshold"],
        color="#555555",
        linestyle="--",
        linewidth=1.2,
        label=f"Best tested F1 threshold ({best_row['threshold']:.2f})",
    )

    axis.set_title("XGBoost Threshold Performance")
    axis.set_xlabel("Classification threshold")
    axis.set_ylabel("Metric value")
    axis.set_xticks(THRESHOLDS)
    axis.set_ylim(0, 1)
    axis.grid(True, linestyle="--", alpha=0.4)
    axis.legend()
    figure.tight_layout()

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)
    LOGGER.info("Saved threshold performance plot to %s", FIGURE_PATH)


def run_threshold_analysis() -> pd.DataFrame:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"XGBoost model not found: {MODEL_PATH}. "
            "Run src/models/train_xgboost.py first."
        )

    x_test, y_test = load_test_data()
    LOGGER.info("Loading trained XGBoost pipeline from %s", MODEL_PATH)
    model = joblib.load(MODEL_PATH)
    predicted_probabilities = model.predict_proba(x_test)[:, 1]

    results = evaluate_thresholds(y_test, predicted_probabilities)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(REPORT_PATH, index=False)
    LOGGER.info("Saved threshold analysis to %s", REPORT_PATH)

    create_performance_plot(results)

    print("\nXGBoost threshold analysis")
    print("--------------------------")
    print(
        results.to_string(
            index=False,
            formatters={
                "threshold": "{:.2f}".format,
                "precision": "{:.6f}".format,
                "recall": "{:.6f}".format,
                "f1": "{:.6f}".format,
                "predicted_default_rate": "{:.6f}".format,
            },
        )
    )
    return results


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    try:
        run_threshold_analysis()
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        LOGGER.error("%s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
