import logging
from pathlib import Path

import pandas as pd


LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

APPLICATION_PATH = RAW_DATA_DIR / "application_train.csv"
FEATURE_PATHS = {
    "bureau": PROCESSED_DATA_DIR / "bureau_features.parquet",
    "previous_application": (
        PROCESSED_DATA_DIR / "previous_application_features.parquet"
    ),
    "installments": PROCESSED_DATA_DIR / "installments_features.parquet",
    "credit_card": PROCESSED_DATA_DIR / "credit_card_features.parquet",
    "pos_cash": PROCESSED_DATA_DIR / "pos_cash_features.parquet",
}
OUTPUT_PATH = PROCESSED_DATA_DIR / "master_pd_dataset.parquet"

KEY_COLUMN = "SK_ID_CURR"
TARGET_COLUMN = "TARGET"


def _validate_input_files() -> None:
    required_paths = [APPLICATION_PATH, *FEATURE_PATHS.values()]
    missing_paths = [path for path in required_paths if not path.is_file()]

    if missing_paths:
        formatted_paths = "\n".join(f"  - {path}" for path in missing_paths)
        raise FileNotFoundError(f"Required input files are missing:\n{formatted_paths}")


def _load_feature_table(name: str, path: Path) -> pd.DataFrame:
    feature_df = pd.read_parquet(path)

    if KEY_COLUMN not in feature_df.columns:
        if feature_df.index.name == KEY_COLUMN:
            feature_df = feature_df.reset_index()
        else:
            raise KeyError(
                f"{name} features do not contain {KEY_COLUMN} as a column or index."
            )

    duplicate_count = int(feature_df[KEY_COLUMN].duplicated().sum())
    if duplicate_count:
        raise ValueError(
            f"{name} features contain {duplicate_count:,} duplicate "
            f"{KEY_COLUMN} values."
        )

    LOGGER.info(
        "Loaded %s features from %s: %s rows, %s columns",
        name,
        path,
        f"{len(feature_df):,}",
        f"{feature_df.shape[1]:,}",
    )
    return feature_df


def _validate_column_overlap(
    master_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    table_name: str,
) -> None:
    overlapping_columns = (
        set(master_df.columns).intersection(feature_df.columns) - {KEY_COLUMN}
    )
    if overlapping_columns:
        columns = ", ".join(sorted(overlapping_columns))
        raise ValueError(
            f"{table_name} features overlap existing columns: {columns}"
        )


def print_validation_report(master_df: pd.DataFrame) -> None:
    duplicate_count = int(master_df[KEY_COLUMN].duplicated().sum())
    target_distribution = master_df[TARGET_COLUMN].value_counts(
        dropna=False, normalize=False
    )
    target_percentage = (
        master_df[TARGET_COLUMN].value_counts(dropna=False, normalize=True) * 100
    )
    missing_percentage = (
        master_df.isna().mean().mul(100).sort_values(ascending=False)
    )
    feature_count = master_df.shape[1] - 2

    print("\nMaster dataset validation")
    print("-------------------------")
    print(f"Shape: {master_df.shape}")
    print(f"Duplicate {KEY_COLUMN}: {duplicate_count:,}")
    print("\nTARGET distribution:")
    for target_value, count in target_distribution.items():
        percentage = target_percentage.loc[target_value]
        print(f"  {target_value}: {count:,} ({percentage:.2f}%)")
    print("\nMissing values by column (%):")
    print(missing_percentage.to_string(float_format=lambda value: f"{value:.2f}"))
    print(f"\nNumber of final features: {feature_count:,}")


def build_master_dataset() -> pd.DataFrame:
    _validate_input_files()

    LOGGER.info("Loading application data from %s", APPLICATION_PATH)
    master_df = pd.read_csv(APPLICATION_PATH)

    required_columns = {KEY_COLUMN, TARGET_COLUMN}
    missing_columns = required_columns - set(master_df.columns)
    if missing_columns:
        raise KeyError(
            "Application data is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    application_duplicates = int(master_df[KEY_COLUMN].duplicated().sum())
    if application_duplicates:
        raise ValueError(
            f"Application data contain {application_duplicates:,} duplicate "
            f"{KEY_COLUMN} values."
        )

    original_row_count = len(master_df)

    for table_name, feature_path in FEATURE_PATHS.items():
        feature_df = _load_feature_table(table_name, feature_path)
        _validate_column_overlap(master_df, feature_df, table_name)
        master_df = master_df.merge(
            feature_df,
            on=KEY_COLUMN,
            how="left",
            validate="one_to_one",
        )

        if len(master_df) != original_row_count:
            raise RuntimeError(
                f"Row count changed after merging {table_name} features."
            )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    master_df.to_parquet(OUTPUT_PATH, index=False)
    LOGGER.info("Saved master dataset to %s", OUTPUT_PATH)

    print_validation_report(master_df)
    return master_df


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    try:
        build_master_dataset()
    except (FileNotFoundError, KeyError, ValueError, RuntimeError) as exc:
        LOGGER.error("%s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
