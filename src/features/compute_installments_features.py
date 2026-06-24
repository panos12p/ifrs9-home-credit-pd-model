from pathlib import Path

import numpy as np
import pandas as pd


def compute_installments_features(
    raw_path: Path,
    out_path: Path,
    filename: str = "installments_payments.csv",
    chunksize: int = 500_000,
):
    raw_file = raw_path / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)

    usecols = [
        "SK_ID_CURR",
        "DAYS_INSTALMENT",
        "DAYS_ENTRY_PAYMENT",
        "AMT_INSTALMENT",
        "AMT_PAYMENT",
    ]
    dtypes = {
        "SK_ID_CURR": "int32",
        "DAYS_INSTALMENT": "float32",
        "DAYS_ENTRY_PAYMENT": "float32",
        "AMT_INSTALMENT": "float32",
        "AMT_PAYMENT": "float32",
    }

    reader = pd.read_csv(
        raw_file,
        encoding="latin1",
        usecols=usecols,
        dtype=dtypes,
        chunksize=chunksize,
    )

    acc = None
    chunk_i = 0
    for chunk in reader:
        chunk_i += 1

        chunk["_days_late"] = chunk["DAYS_ENTRY_PAYMENT"] - chunk["DAYS_INSTALMENT"]
        chunk["_is_late"] = (chunk["_days_late"] > 0).astype("int8")

        agg = chunk.groupby("SK_ID_CURR", observed=True).agg(
            installment_count=("AMT_INSTALMENT", "count"),
            installment_total_instalment=("AMT_INSTALMENT", "sum"),
            cnt_instalment=("AMT_INSTALMENT", "count"),
            installment_total_payment=("AMT_PAYMENT", "sum"),
            cnt_payment=("AMT_PAYMENT", "count"),
            installment_max_payment=("AMT_PAYMENT", "max"),
            sum_days_late=("_days_late", "sum"),
            cnt_days_late=("_days_late", "count"),
            installment_days_late_max=("_days_late", "max"),
            installment_late_payment_count=("_is_late", "sum"),
        )

        count_cols = [
            "installment_count",
            "cnt_instalment",
            "cnt_payment",
            "cnt_days_late",
            "installment_late_payment_count",
        ]
        for col in count_cols:
            agg[col] = agg[col].astype("int32")

        if acc is None:
            acc = agg
        else:
            acc = acc.join(agg, how="outer", rsuffix="_new")

            add_cols = [
                "installment_count",
                "installment_total_instalment",
                "cnt_instalment",
                "installment_total_payment",
                "cnt_payment",
                "sum_days_late",
                "cnt_days_late",
                "installment_late_payment_count",
            ]
            for col in add_cols:
                acc[col] = acc[col].fillna(0) + acc[f"{col}_new"].fillna(0)

            acc["installment_max_payment"] = pd.concat(
                [acc["installment_max_payment"], acc["installment_max_payment_new"]],
                axis=1,
            ).max(axis=1)
            acc["installment_days_late_max"] = pd.concat(
                [
                    acc["installment_days_late_max"],
                    acc["installment_days_late_max_new"],
                ],
                axis=1,
            ).max(axis=1)

            drop_cols = [col for col in acc.columns if col.endswith("_new")]
            acc = acc.drop(columns=drop_cols)

        if chunk_i % 5 == 0:
            print(f"Processed {chunk_i} chunks, accumulated {len(acc)} borrowers")

    if acc is None:
        print("No data processed.")
        return

    valid_instalment_total = acc["installment_total_instalment"].gt(0) & acc[
        "installment_total_instalment"
    ].notna()
    acc["installment_payment_ratio"] = np.nan
    acc.loc[valid_instalment_total, "installment_payment_ratio"] = (
        acc.loc[valid_instalment_total, "installment_total_payment"]
        / acc.loc[valid_instalment_total, "installment_total_instalment"]
    )
    acc["installment_payment_ratio"] = acc["installment_payment_ratio"].astype("float32")
    acc["installment_underpayment_flag"] = (
        acc["installment_payment_ratio"] < 1
    ).astype("int8")
    acc["installment_mean_payment"] = (
        acc["installment_total_payment"] / acc["cnt_payment"].replace(0, np.nan)
    ).astype("float32")
    acc["installment_mean_instalment"] = (
        acc["installment_total_instalment"] / acc["cnt_instalment"].replace(0, np.nan)
    ).astype("float32")
    acc["installment_days_late_mean"] = (
        acc["sum_days_late"] / acc["cnt_days_late"].replace(0, np.nan)
    ).astype("float32")
    acc["installment_late_payment_rate"] = (
        acc["installment_late_payment_count"]
        / acc["installment_count"].replace(0, np.nan)
    ).astype("float32")

    out_df = acc[
        [
            "installment_count",
            "installment_total_instalment",
            "installment_total_payment",
            "installment_payment_ratio",
            "installment_underpayment_flag",
            "installment_mean_payment",
            "installment_max_payment",
            "installment_mean_instalment",
            "installment_days_late_mean",
            "installment_days_late_max",
            "installment_late_payment_count",
            "installment_late_payment_rate",
        ]
    ].copy()

    count_cols = [
        "installment_count",
        "installment_underpayment_flag",
        "installment_late_payment_count",
    ]
    for col in count_cols:
        out_df[col] = out_df[col].astype("int32")

    float_cols = [col for col in out_df.columns if col not in count_cols]
    out_df[float_cols] = out_df[float_cols].astype("float32")

    out_df.index.name = "SK_ID_CURR"
    out_df.to_parquet(out_path, index=True)
    print(f"Wrote installments features to {out_path} (rows={len(out_df)})")


if __name__ == "__main__":
    RAW = Path(r"E:\ifrs9-home-credit-pd-model\data\raw")
    OUT = Path(r"E:\ifrs9-home-credit-pd-model\data\processed\installments_features.parquet")
    compute_installments_features(RAW, OUT)
