from pathlib import Path
import pandas as pd
import numpy as np


def compute_bureau_features(
    raw_path: Path,
    out_path: Path,
    filename: str = "bureau.csv",
    chunksize: int = 200_000,
):
    raw_file = raw_path / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)

    reader = pd.read_csv(raw_file, encoding="latin1", chunksize=chunksize)

    acc = None
    chunk_i = 0
    for chunk in reader:
        chunk_i += 1
        # normalize columns we will use
        # ensure numeric types
        for col in ["AMT_CREDIT_SUM", "AMT_ANNUITY", "DAYS_CREDIT"]:
            if col in chunk.columns:
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce").astype("float32")

        # boolean flags for active/closed
        if "CREDIT_ACTIVE" in chunk.columns:
            ca = chunk["CREDIT_ACTIVE"].astype(str).str.lower()
            chunk["_is_active"] = (ca == "active").astype("int8")
            chunk["_is_closed"] = (ca == "closed").astype("int8")
        else:
            chunk["_is_active"] = 0
            chunk["_is_closed"] = 0

        # groupby SK_ID_CURR
        agg = chunk.groupby("SK_ID_CURR").agg(
            bureau_loan_count=("SK_ID_BUREAU", "count"),
            bureau_active_loan_count=("_is_active", "sum"),
            bureau_closed_loan_count=("_is_closed", "sum"),
            bureau_total_credit_sum=("AMT_CREDIT_SUM", "sum"),
            cnt_credit_sum=("AMT_CREDIT_SUM", "count"),
            bureau_max_credit_sum=("AMT_CREDIT_SUM", "max"),
            sum_annuity=("AMT_ANNUITY", "sum"),
            cnt_annuity=("AMT_ANNUITY", "count"),
            sum_days_credit=("DAYS_CREDIT", "sum"),
            cnt_days_credit=("DAYS_CREDIT", "count"),
            bureau_days_credit_min=("DAYS_CREDIT", "min"),
            bureau_days_credit_max=("DAYS_CREDIT", "max"),
        )

        # cast small ints
        for c in ["bureau_loan_count", "bureau_active_loan_count", "bureau_closed_loan_count"]:
            if c in agg.columns:
                agg[c] = agg[c].astype("int32")

        if acc is None:
            acc = agg
        else:
            # merge and combine numerics
            acc = acc.join(agg, how="outer", rsuffix="_new")
            # sum counts and sums
            add_cols = [
                "bureau_loan_count",
                "bureau_active_loan_count",
                "bureau_closed_loan_count",
                "bureau_total_credit_sum",
                "cnt_credit_sum",
                "sum_annuity",
                "cnt_annuity",
                "sum_days_credit",
                "cnt_days_credit",
            ]
            for col in add_cols:
                a = acc[col].fillna(0)
                b = acc.get(col + "_new", 0).fillna(0)
                acc[col] = a + b
            # max / min merges
            acc["bureau_max_credit_sum"] = pd.concat(
                [acc["bureau_max_credit_sum"], acc.get("bureau_max_credit_sum_new", pd.Series(dtype="float32"))],
                axis=1,
            ).max(axis=1)
            acc["bureau_days_credit_min"] = pd.concat(
                [acc["bureau_days_credit_min"], acc.get("bureau_days_credit_min_new", pd.Series(dtype="float32"))],
                axis=1,
            ).min(axis=1)
            acc["bureau_days_credit_max"] = pd.concat(
                [acc["bureau_days_credit_max"], acc.get("bureau_days_credit_max_new", pd.Series(dtype="float32"))],
                axis=1,
            ).max(axis=1)

            # drop the _new columns
            drop_cols = [c for c in acc.columns if c.endswith("_new")]
            if drop_cols:
                acc = acc.drop(columns=drop_cols)

        if chunk_i % 5 == 0:
            print(f"Processed {chunk_i} chunks, accumulated {len(acc)} borrowers")

    # finalize derived metrics
    if acc is None:
        print("No data processed.")
        return

    # compute means
    acc["bureau_mean_credit_sum"] = (
        acc["bureau_total_credit_sum"] / acc["cnt_credit_sum"].replace(0, np.nan)
    ).astype("float32")
    acc["bureau_mean_annuity"] = (acc["sum_annuity"] / acc["cnt_annuity"].replace(0, np.nan)).astype(
        "float32"
    )
    acc["bureau_days_credit_mean"] = (
        acc["sum_days_credit"] / acc["cnt_days_credit"].replace(0, np.nan)
    ).astype("float32")

    # select final columns
    out_df = acc[
        [
            "bureau_loan_count",
            "bureau_active_loan_count",
            "bureau_closed_loan_count",
            "bureau_total_credit_sum",
            "bureau_mean_credit_sum",
            "bureau_max_credit_sum",
            "bureau_mean_annuity",
            "bureau_days_credit_mean",
            "bureau_days_credit_min",
            "bureau_days_credit_max",
        ]
    ].copy()

    # write parquet
    out_file = out_path
    out_df.index.name = "SK_ID_CURR"
    out_df.to_parquet(out_file, index=True)
    print(f"Wrote bureau features to {out_file} (rows={len(out_df)})")


if __name__ == "__main__":
    RAW = Path(r"E:\ifrs9-home-credit-pd-model\data\raw")
    OUT = Path(r"E:\ifrs9-home-credit-pd-model\data\processed\bureau_features.parquet")
    compute_bureau_features(RAW, OUT)
