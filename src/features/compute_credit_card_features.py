from pathlib import Path

import numpy as np
import pandas as pd


def compute_credit_card_features(
    raw_path: Path,
    out_path: Path,
    filename: str = "credit_card_balance.csv",
    chunksize: int = 500_000,
):
    raw_file = raw_path / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)

    usecols = [
        "SK_ID_CURR",
        "AMT_BALANCE",
        "AMT_CREDIT_LIMIT_ACTUAL",
        "AMT_DRAWINGS_CURRENT",
        "AMT_PAYMENT_TOTAL_CURRENT",
        "SK_DPD",
    ]
    dtypes = {
        "SK_ID_CURR": "int32",
        "AMT_BALANCE": "float32",
        "AMT_CREDIT_LIMIT_ACTUAL": "float32",
        "AMT_DRAWINGS_CURRENT": "float32",
        "AMT_PAYMENT_TOTAL_CURRENT": "float32",
        "SK_DPD": "float32",
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

        valid_limit = chunk["AMT_CREDIT_LIMIT_ACTUAL"].gt(0) & chunk[
            "AMT_CREDIT_LIMIT_ACTUAL"
        ].notna()
        chunk["_utilization"] = np.nan
        chunk.loc[valid_limit, "_utilization"] = (
            chunk.loc[valid_limit, "AMT_BALANCE"]
            / chunk.loc[valid_limit, "AMT_CREDIT_LIMIT_ACTUAL"]
        )
        chunk["_is_late"] = (chunk["SK_DPD"] > 0).astype("int8")

        agg = chunk.groupby("SK_ID_CURR", observed=True).agg(
            credit_card_month_count=("AMT_BALANCE", "count"),
            sum_balance=("AMT_BALANCE", "sum"),
            cnt_balance=("AMT_BALANCE", "count"),
            credit_card_max_balance=("AMT_BALANCE", "max"),
            sum_credit_limit=("AMT_CREDIT_LIMIT_ACTUAL", "sum"),
            cnt_credit_limit=("AMT_CREDIT_LIMIT_ACTUAL", "count"),
            credit_card_max_credit_limit=("AMT_CREDIT_LIMIT_ACTUAL", "max"),
            sum_utilization=("_utilization", "sum"),
            cnt_utilization=("_utilization", "count"),
            credit_card_max_utilization=("_utilization", "max"),
            credit_card_total_drawings=("AMT_DRAWINGS_CURRENT", "sum"),
            sum_payment_total=("AMT_PAYMENT_TOTAL_CURRENT", "sum"),
            cnt_payment_total=("AMT_PAYMENT_TOTAL_CURRENT", "count"),
            credit_card_late_count=("_is_late", "sum"),
            credit_card_max_dpd=("SK_DPD", "max"),
        )

        count_cols = [
            "credit_card_month_count",
            "cnt_balance",
            "cnt_credit_limit",
            "cnt_utilization",
            "cnt_payment_total",
            "credit_card_late_count",
        ]
        for col in count_cols:
            agg[col] = agg[col].astype("int32")

        if acc is None:
            acc = agg
        else:
            acc = acc.join(agg, how="outer", rsuffix="_new")

            add_cols = [
                "credit_card_month_count",
                "sum_balance",
                "cnt_balance",
                "sum_credit_limit",
                "cnt_credit_limit",
                "sum_utilization",
                "cnt_utilization",
                "credit_card_total_drawings",
                "sum_payment_total",
                "cnt_payment_total",
                "credit_card_late_count",
            ]
            for col in add_cols:
                acc[col] = acc[col].fillna(0) + acc[f"{col}_new"].fillna(0)

            max_cols = [
                "credit_card_max_balance",
                "credit_card_max_credit_limit",
                "credit_card_max_utilization",
                "credit_card_max_dpd",
            ]
            for col in max_cols:
                acc[col] = pd.concat([acc[col], acc[f"{col}_new"]], axis=1).max(axis=1)

            drop_cols = [col for col in acc.columns if col.endswith("_new")]
            acc = acc.drop(columns=drop_cols)

        if chunk_i % 5 == 0:
            print(f"Processed {chunk_i} chunks, accumulated {len(acc)} borrowers")

    if acc is None:
        print("No data processed.")
        return

    acc["credit_card_mean_balance"] = (
        acc["sum_balance"] / acc["cnt_balance"].replace(0, np.nan)
    ).astype("float32")
    acc["credit_card_mean_credit_limit"] = (
        acc["sum_credit_limit"] / acc["cnt_credit_limit"].replace(0, np.nan)
    ).astype("float32")
    acc["credit_card_mean_utilization"] = (
        acc["sum_utilization"] / acc["cnt_utilization"].replace(0, np.nan)
    ).astype("float32")
    acc["credit_card_mean_payment_total"] = (
        acc["sum_payment_total"] / acc["cnt_payment_total"].replace(0, np.nan)
    ).astype("float32")
    acc["credit_card_late_rate"] = (
        acc["credit_card_late_count"] / acc["credit_card_month_count"].replace(0, np.nan)
    ).astype("float32")

    out_df = acc[
        [
            "credit_card_month_count",
            "credit_card_mean_balance",
            "credit_card_max_balance",
            "credit_card_mean_credit_limit",
            "credit_card_max_credit_limit",
            "credit_card_mean_utilization",
            "credit_card_max_utilization",
            "credit_card_total_drawings",
            "credit_card_mean_payment_total",
            "credit_card_late_count",
            "credit_card_late_rate",
            "credit_card_max_dpd",
        ]
    ].copy()

    count_cols = ["credit_card_month_count", "credit_card_late_count"]
    for col in count_cols:
        out_df[col] = out_df[col].astype("int32")

    float_cols = [col for col in out_df.columns if col not in count_cols]
    out_df[float_cols] = out_df[float_cols].astype("float32")

    out_df.index.name = "SK_ID_CURR"
    out_df.to_parquet(out_path, index=True)
    print(f"Wrote credit card features to {out_path} (rows={len(out_df)})")


if __name__ == "__main__":
    RAW = Path(r"E:\ifrs9-home-credit-pd-model\data\raw")
    OUT = Path(r"E:\ifrs9-home-credit-pd-model\data\processed\credit_card_features.parquet")
    compute_credit_card_features(RAW, OUT)
