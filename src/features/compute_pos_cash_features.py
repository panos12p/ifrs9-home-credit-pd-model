import logging
from pathlib import Path

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "pos_cash_features.parquet"

KEY_COLUMN = "SK_ID_CURR"


def _combine_aggregates(
    accumulated: pd.DataFrame | None,
    current: pd.DataFrame,
) -> pd.DataFrame:
    if accumulated is None:
        return current

    combined = accumulated.join(current, how="outer", rsuffix="_new")

    additive_columns = [
        "pos_cash_month_count",
        "sum_instalment",
        "count_instalment",
        "sum_future_instalments",
        "count_future_instalments",
        "pos_cash_late_count",
    ]
    for column in additive_columns:
        combined[column] = (
            combined[column].fillna(0) + combined[f"{column}_new"].fillna(0)
        )

    maximum_columns = [
        "pos_cash_max_instalment",
        "pos_cash_max_future_instalments",
        "pos_cash_max_dpd",
        "pos_cash_max_dpd_def",
    ]
    for column in maximum_columns:
        combined[column] = combined[
            [column, f"{column}_new"]
        ].max(axis=1, skipna=True)

    return combined.drop(
        columns=[column for column in combined.columns if column.endswith("_new")]
    )


def _print_validation_report(features: pd.DataFrame) -> None:
    duplicate_count = int(features.index.duplicated().sum())
    invalid_late_rates = int(
        (~features["pos_cash_late_rate"].between(0, 1, inclusive="both"))
        .fillna(False)
        .sum()
    )
    missing_percentage = (
        features.isna().mean().mul(100).sort_values(ascending=False)
    )

    print("\nPOS cash feature validation")
    print("---------------------------")
    print(f"Shape: {features.shape}")
    print(f"Duplicate {KEY_COLUMN}: {duplicate_count:,}")
    print(f"Late rates outside [0, 1]: {invalid_late_rates:,}")
    print("\nMissing values by column (%):")
    print(missing_percentage.to_string(float_format=lambda value: f"{value:.2f}"))


def compute_pos_cash_features(
    raw_path: Path = RAW_DATA_DIR,
    out_path: Path = OUTPUT_PATH,
    filename: str = "POS_CASH_balance.csv",
    chunksize: int = 500_000,
) -> pd.DataFrame:
    raw_file = raw_path / filename
    if not raw_file.is_file():
        raise FileNotFoundError(f"POS cash source file not found: {raw_file}")

    if chunksize <= 0:
        raise ValueError("chunksize must be greater than zero.")

    usecols = [
        "SK_ID_PREV",
        KEY_COLUMN,
        "CNT_INSTALMENT",
        "CNT_INSTALMENT_FUTURE",
        "SK_DPD",
        "SK_DPD_DEF",
    ]
    dtypes = {
        "SK_ID_PREV": "int32",
        KEY_COLUMN: "int32",
        "CNT_INSTALMENT": "float32",
        "CNT_INSTALMENT_FUTURE": "float32",
        "SK_DPD": "float32",
        "SK_DPD_DEF": "float32",
    }

    LOGGER.info("Reading POS cash data from %s", raw_file)
    reader = pd.read_csv(
        raw_file,
        usecols=usecols,
        dtype=dtypes,
        chunksize=chunksize,
    )

    accumulated = None
    contract_pairs = []

    for chunk_number, chunk in enumerate(reader, start=1):
        chunk["_is_late"] = chunk["SK_DPD"].gt(0).astype("int8")

        current = chunk.groupby(KEY_COLUMN, observed=True).agg(
            pos_cash_month_count=("SK_ID_PREV", "size"),
            sum_instalment=("CNT_INSTALMENT", "sum"),
            count_instalment=("CNT_INSTALMENT", "count"),
            pos_cash_max_instalment=("CNT_INSTALMENT", "max"),
            sum_future_instalments=("CNT_INSTALMENT_FUTURE", "sum"),
            count_future_instalments=("CNT_INSTALMENT_FUTURE", "count"),
            pos_cash_max_future_instalments=("CNT_INSTALMENT_FUTURE", "max"),
            pos_cash_late_count=("_is_late", "sum"),
            pos_cash_max_dpd=("SK_DPD", "max"),
            pos_cash_max_dpd_def=("SK_DPD_DEF", "max"),
        )
        accumulated = _combine_aggregates(accumulated, current)

        contract_pairs.append(
            chunk[[KEY_COLUMN, "SK_ID_PREV"]].drop_duplicates()
        )

        if chunk_number % 5 == 0:
            LOGGER.info(
                "Processed %s chunks; accumulated %s borrowers",
                chunk_number,
                f"{len(accumulated):,}",
            )

    if accumulated is None:
        raise ValueError(f"No rows were read from {raw_file}.")

    unique_contracts = pd.concat(contract_pairs, ignore_index=True).drop_duplicates()
    contract_counts = unique_contracts.groupby(KEY_COLUMN, observed=True)[
        "SK_ID_PREV"
    ].nunique()
    accumulated["pos_cash_contract_count"] = contract_counts

    accumulated["pos_cash_mean_instalment"] = (
        accumulated["sum_instalment"]
        / accumulated["count_instalment"].replace(0, np.nan)
    )
    accumulated["pos_cash_mean_future_instalments"] = (
        accumulated["sum_future_instalments"]
        / accumulated["count_future_instalments"].replace(0, np.nan)
    )
    accumulated["pos_cash_late_rate"] = (
        accumulated["pos_cash_late_count"]
        / accumulated["pos_cash_month_count"].replace(0, np.nan)
    )

    output_columns = [
        "pos_cash_month_count",
        "pos_cash_contract_count",
        "pos_cash_mean_instalment",
        "pos_cash_max_instalment",
        "pos_cash_mean_future_instalments",
        "pos_cash_max_future_instalments",
        "pos_cash_late_count",
        "pos_cash_late_rate",
        "pos_cash_max_dpd",
        "pos_cash_max_dpd_def",
    ]
    features = accumulated[output_columns].copy()

    integer_columns = [
        "pos_cash_month_count",
        "pos_cash_contract_count",
        "pos_cash_late_count",
    ]
    features[integer_columns] = features[integer_columns].astype("int32")
    float_columns = [
        column for column in output_columns if column not in integer_columns
    ]
    features[float_columns] = features[float_columns].astype("float32")
    features.index.name = KEY_COLUMN
    features = features.sort_index()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out_path, index=True)
    LOGGER.info(
        "Saved POS cash features to %s: %s rows, %s columns",
        out_path,
        f"{features.shape[0]:,}",
        features.shape[1],
    )

    _print_validation_report(features)
    return features


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    try:
        compute_pos_cash_features()
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        LOGGER.error("%s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
