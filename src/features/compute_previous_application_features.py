from pathlib import Path

import numpy as np
import pandas as pd


def compute_previous_application_features(
    raw_path: Path,
    out_path: Path,
    filename: str = "previous_application.csv",
    chunksize: int = 300_000,
):
    raw_file = raw_path / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)

    usecols = [
        "SK_ID_PREV",
        "SK_ID_CURR",
        "NAME_CONTRACT_STATUS",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_APPLICATION",
        "DAYS_DECISION",
    ]
    dtypes = {
        "SK_ID_PREV": "int32",
        "SK_ID_CURR": "int32",
        "NAME_CONTRACT_STATUS": "category",
        "AMT_CREDIT": "float32",
        "AMT_ANNUITY": "float32",
        "AMT_APPLICATION": "float32",
        "DAYS_DECISION": "float32",
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

        status = chunk["NAME_CONTRACT_STATUS"].astype("string").str.lower()
        chunk["_is_approved"] = (status == "approved").astype("int8")
        chunk["_is_refused"] = (status == "refused").astype("int8")

        agg = chunk.groupby("SK_ID_CURR", observed=True).agg(
            prev_application_count=("SK_ID_PREV", "count"),
            prev_approved_count=("_is_approved", "sum"),
            prev_refused_count=("_is_refused", "sum"),
            sum_credit=("AMT_CREDIT", "sum"),
            cnt_credit=("AMT_CREDIT", "count"),
            prev_max_credit=("AMT_CREDIT", "max"),
            sum_annuity=("AMT_ANNUITY", "sum"),
            cnt_annuity=("AMT_ANNUITY", "count"),
            sum_application_amount=("AMT_APPLICATION", "sum"),
            cnt_application_amount=("AMT_APPLICATION", "count"),
            prev_recent_application_days=("DAYS_DECISION", "max"),
        )

        count_cols = [
            "prev_application_count",
            "prev_approved_count",
            "prev_refused_count",
            "cnt_credit",
            "cnt_annuity",
            "cnt_application_amount",
        ]
        for col in count_cols:
            agg[col] = agg[col].astype("int32")

        if acc is None:
            acc = agg
        else:
            acc = acc.join(agg, how="outer", rsuffix="_new")

            add_cols = [
                "prev_application_count",
                "prev_approved_count",
                "prev_refused_count",
                "sum_credit",
                "cnt_credit",
                "sum_annuity",
                "cnt_annuity",
                "sum_application_amount",
                "cnt_application_amount",
            ]
            for col in add_cols:
                acc[col] = acc[col].fillna(0) + acc[f"{col}_new"].fillna(0)

            acc["prev_max_credit"] = pd.concat(
                [acc["prev_max_credit"], acc["prev_max_credit_new"]],
                axis=1,
            ).max(axis=1)
            acc["prev_recent_application_days"] = pd.concat(
                [
                    acc["prev_recent_application_days"],
                    acc["prev_recent_application_days_new"],
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

    acc["prev_approval_rate"] = (
        acc["prev_approved_count"] / acc["prev_application_count"].replace(0, np.nan)
    ).astype("float32")
    acc["prev_mean_credit"] = (
        acc["sum_credit"] / acc["cnt_credit"].replace(0, np.nan)
    ).astype("float32")
    acc["prev_mean_annuity"] = (
        acc["sum_annuity"] / acc["cnt_annuity"].replace(0, np.nan)
    ).astype("float32")
    acc["prev_mean_application_amount"] = (
        acc["sum_application_amount"] / acc["cnt_application_amount"].replace(0, np.nan)
    ).astype("float32")

    out_df = acc[
        [
            "prev_application_count",
            "prev_approved_count",
            "prev_refused_count",
            "prev_approval_rate",
            "prev_mean_credit",
            "prev_max_credit",
            "prev_mean_annuity",
            "prev_mean_application_amount",
            "prev_recent_application_days",
        ]
    ].copy()

    count_cols = [
        "prev_application_count",
        "prev_approved_count",
        "prev_refused_count",
    ]
    for col in count_cols:
        out_df[col] = out_df[col].astype("int32")

    float_cols = [col for col in out_df.columns if col not in count_cols]
    out_df[float_cols] = out_df[float_cols].astype("float32")

    out_df.index.name = "SK_ID_CURR"
    out_df.to_parquet(out_path, index=True)
    print(f"Wrote previous application features to {out_path} (rows={len(out_df)})")


if __name__ == "__main__":
    RAW = Path(r"E:\ifrs9-home-credit-pd-model\data\raw")
    OUT = Path(
        r"E:\ifrs9-home-credit-pd-model\data\processed\previous_application_features.parquet"
    )
    compute_previous_application_features(RAW, OUT)
