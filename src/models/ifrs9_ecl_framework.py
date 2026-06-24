import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "master_pd_dataset.parquet"
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_pd_model.joblib"
REPORT_PATH = PROJECT_ROOT / "outputs" / "reports" / "ifrs9_ecl_report.txt"
BORROWER_OUTPUT_PATH = (
    PROJECT_ROOT / "outputs" / "reports" / "ifrs9_borrower_ecl.parquet"
)

ID_COLUMN = "SK_ID_CURR"
TARGET_COLUMN = "TARGET"
CONTRACT_COLUMN = "NAME_CONTRACT_TYPE"
CREDIT_COLUMN = "AMT_CREDIT"

STAGE_2_PD_THRESHOLD = 0.15
CASH_LOAN_LGD = 0.45
REVOLVING_LOAN_LGD = 0.55
CASH_LOAN_EAD_FACTOR = 1.00
REVOLVING_LOAN_EAD_FACTOR = 0.75


def load_inputs() -> tuple[pd.DataFrame, object]:
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Master dataset not found: {DATASET_PATH}")
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"XGBoost model not found: {MODEL_PATH}. "
            "Run src/models/train_xgboost.py first."
        )

    LOGGER.info("Loading master dataset from %s", DATASET_PATH)
    dataset = pd.read_parquet(DATASET_PATH)

    required_columns = {
        ID_COLUMN,
        TARGET_COLUMN,
        CONTRACT_COLUMN,
        CREDIT_COLUMN,
    }
    missing_columns = required_columns - set(dataset.columns)
    if missing_columns:
        raise KeyError(
            "Master dataset is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )
    if dataset[ID_COLUMN].duplicated().any():
        duplicate_count = int(dataset[ID_COLUMN].duplicated().sum())
        raise ValueError(
            f"Master dataset contains {duplicate_count:,} duplicate "
            f"{ID_COLUMN} values."
        )
    if dataset[CREDIT_COLUMN].isna().any():
        missing_count = int(dataset[CREDIT_COLUMN].isna().sum())
        raise ValueError(
            f"{CREDIT_COLUMN} contains {missing_count:,} missing values."
        )
    if dataset[CREDIT_COLUMN].le(0).any():
        invalid_count = int(dataset[CREDIT_COLUMN].le(0).sum())
        raise ValueError(
            f"{CREDIT_COLUMN} contains {invalid_count:,} non-positive values."
        )

    LOGGER.info("Loading trained XGBoost pipeline from %s", MODEL_PATH)
    model = joblib.load(MODEL_PATH)
    return dataset, model


def generate_borrower_ecl(
    dataset: pd.DataFrame,
    model,
) -> pd.DataFrame:
    model_features = dataset.drop(columns=[ID_COLUMN, TARGET_COLUMN])
    predicted_pd = model.predict_proba(model_features)[:, 1]

    if not np.isfinite(predicted_pd).all():
        raise ValueError("Model generated non-finite PD values.")
    if ((predicted_pd < 0) | (predicted_pd > 1)).any():
        raise ValueError("Model generated PD values outside [0, 1].")

    is_revolving = dataset[CONTRACT_COLUMN].eq("Revolving loans")
    lgd = np.where(is_revolving, REVOLVING_LOAN_LGD, CASH_LOAN_LGD)
    ead_factor = np.where(
        is_revolving,
        REVOLVING_LOAN_EAD_FACTOR,
        CASH_LOAN_EAD_FACTOR,
    )
    ead = dataset[CREDIT_COLUMN].to_numpy(dtype="float64") * ead_factor
    stage = np.where(
        predicted_pd >= STAGE_2_PD_THRESHOLD,
        "Stage 2",
        "Stage 1",
    )
    ecl = predicted_pd * lgd * ead

    borrower_ecl = pd.DataFrame(
        {
            ID_COLUMN: dataset[ID_COLUMN].to_numpy(),
            "contract_type": dataset[CONTRACT_COLUMN].to_numpy(),
            "predicted_pd": predicted_pd,
            "lgd": lgd,
            "ead": ead,
            "stage": stage,
            "expected_credit_loss": ecl,
        }
    )

    if borrower_ecl["expected_credit_loss"].lt(0).any():
        raise ValueError("Calculated ECL contains negative values.")

    return borrower_ecl


def create_stage_summary(borrower_ecl: pd.DataFrame) -> pd.DataFrame:
    summary = (
        borrower_ecl.groupby("stage", observed=True)
        .agg(
            borrower_count=(ID_COLUMN, "count"),
            average_pd=("predicted_pd", "mean"),
            average_lgd=("lgd", "mean"),
            total_ead=("ead", "sum"),
            total_ecl=("expected_credit_loss", "sum"),
        )
        .reindex(["Stage 1", "Stage 2"])
        .reset_index()
    )
    summary["borrower_share"] = (
        summary["borrower_count"] / summary["borrower_count"].sum()
    )
    summary["ecl_rate"] = summary["total_ecl"] / summary["total_ead"]
    return summary


def create_report(
    borrower_ecl: pd.DataFrame,
    stage_summary: pd.DataFrame,
) -> str:
    total_ead = borrower_ecl["ead"].sum()
    total_ecl = borrower_ecl["expected_credit_loss"].sum()
    exposure_weighted_pd = np.average(
        borrower_ecl["predicted_pd"],
        weights=borrower_ecl["ead"],
    )
    exposure_weighted_lgd = np.average(
        borrower_ecl["lgd"],
        weights=borrower_ecl["ead"],
    )

    lines = [
        "Simplified IFRS 9 ECL Framework Report",
        "=====================================",
        "",
        "Scope",
        "-----",
        f"Borrowers scored: {len(borrower_ecl):,}",
        "PD model: trained XGBoost classifier",
        f"Source dataset: {DATASET_PATH}",
        "",
        "Simplified assumptions",
        "----------------------",
        f"Stage 1: predicted PD below {STAGE_2_PD_THRESHOLD:.0%}",
        f"Stage 2: predicted PD at or above {STAGE_2_PD_THRESHOLD:.0%}",
        f"Cash-loan LGD: {CASH_LOAN_LGD:.0%}",
        f"Revolving-loan LGD: {REVOLVING_LOAN_LGD:.0%}",
        f"Cash-loan EAD: {CASH_LOAN_EAD_FACTOR:.0%} of AMT_CREDIT",
        "Revolving-loan EAD: "
        f"{REVOLVING_LOAN_EAD_FACTOR:.0%} of AMT_CREDIT",
        "ECL formula: PD x LGD x EAD",
        "",
        "Portfolio summary",
        "-----------------",
        f"Total EAD: {total_ead:,.2f}",
        f"Total ECL: {total_ecl:,.2f}",
        f"Portfolio ECL rate: {total_ecl / total_ead:.2%}",
        f"Exposure-weighted PD: {exposure_weighted_pd:.2%}",
        f"Exposure-weighted LGD: {exposure_weighted_lgd:.2%}",
        "",
        "Stage summary",
        "-------------",
        (
            "Stage      Borrowers   Share    Average PD   Average LGD"
            "          Total EAD          Total ECL   ECL Rate"
        ),
    ]

    for row in stage_summary.itertuples(index=False):
        lines.append(
            f"{row.stage:<10}"
            f"{row.borrower_count:>10,d}   "
            f"{row.borrower_share:>6.2%}   "
            f"{row.average_pd:>10.2%}   "
            f"{row.average_lgd:>11.2%}   "
            f"{row.total_ead:>16,.2f}   "
            f"{row.total_ecl:>16,.2f}   "
            f"{row.ecl_rate:>8.2%}"
        )

    lines.extend(
        [
            "",
            "Important limitations",
            "---------------------",
            (
                "This is an analytical proxy, not a production IFRS 9 "
                "measurement framework."
            ),
            (
                "The model output is treated as a single-period PD and is not "
                "a calibrated 12-month or lifetime PD term structure."
            ),
            (
                "Stage 2 uses a fixed PD threshold rather than a relative "
                "increase in credit risk since initial recognition."
            ),
            (
                "LGD and EAD are rule-based assumptions because contractual "
                "cash flows, recoveries, collateral values, and credit "
                "conversion factors are unavailable."
            ),
            (
                "The framework does not include Stage 3, discounting, "
                "forward-looking macroeconomic scenarios, or scenario weights."
            ),
            "",
            f"Borrower-level output: {BORROWER_OUTPUT_PATH}",
            "",
        ]
    )
    return "\n".join(lines)


def run_ecl_framework() -> pd.DataFrame:
    dataset, model = load_inputs()
    LOGGER.info("Generating borrower-level PD and ECL estimates")
    borrower_ecl = generate_borrower_ecl(dataset, model)
    stage_summary = create_stage_summary(borrower_ecl)
    report = create_report(borrower_ecl, stage_summary)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    borrower_ecl.to_parquet(BORROWER_OUTPUT_PATH, index=False)
    REPORT_PATH.write_text(report, encoding="utf-8")

    LOGGER.info("Saved borrower-level ECL output to %s", BORROWER_OUTPUT_PATH)
    LOGGER.info("Saved portfolio ECL report to %s", REPORT_PATH)
    print(report)
    return borrower_ecl


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    try:
        run_ecl_framework()
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        LOGGER.error("%s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
