"""
Part 2: Automated Data Quality Audit & Point-in-Time Panel Construction
=======================================================================
1. Audits all 30 Call Report fields across 1,682,738 bank-quarters (1984Q1-2026Q2):
   - Missingness rates overall and by decade (1980s-2020s).
   - Structural zero encoding in Basel capital ratios (RBCRWAJ=0 pre-1990 Basel I
     and post-2020 CBLR framework; RBC1AAJ=0 for 1984-1989 TFR thrifts).
   - Economically valid negatives (losses/insolvency: EQ, NETINC, ROA, ROE) vs.
     accounting impossibilities (negative balance-sheet stocks/flows).
   - Accounting identity checks, reporting frequency gaps, and bank exit types
     (failures.parquet vs. institutions.parquet ACTIVE & CHANGEC1 codes).
2. Applies strict point-in-time cleaning rules (zero look-ahead bias):
   - Excludes post-failure conservatorship filings (rep_dt >= outright fail_dt).
   - Converts structural sentinel zeros in RBCRWAJ/RBC1AAJ (when EQ != 0) to null
     and records explicit binary validity/missingness flags.
   - Replaces impossible negative stock/flow values with null prior to causal
     1-quarter forward-fill strictly within contiguous active reporting spells.
   - Caps extreme ratio outliers (ROA, ROE, NIMY, RBC1AAJ, RBCRWAJ) using bounds
     estimated strictly on the Development period (<= 2006Q4).
3. Constructs exact 4-quarter forward failure labels y_{i,q} in (q, q + 4 quarters]
   for quarters <= 2025Q2 (excluding right-censored 2025Q3-2026Q2) with a 4-quarter
   embargo gap (2007Q1-2007Q4) between Development (1984Q1-2006Q4) and Test (2008Q1-2025Q2).
4. Saves clean panel to data_clean/panel_labeled.parquet and audit tables/charts to outputs/.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import pyarrow as pa

SEED = 42
RAW_DIR = ROOT / "data_raw" / "fdic"
FIN_DIR = RAW_DIR / "financials"
CLEAN_DIR = ROOT / "data_clean"
OUT_DIR = ROOT / "outputs"
CLEAN_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

NUMERIC_COLS = (
    "ASSET", "DEP", "DEPDOM", "EQ", "BRO", "CHBAL", "SC",
    "LNLSGR", "LNLSNET", "LNRE", "LNCI", "LNCON", "LNATRES",
    "NCLNLS", "P3ASSET", "P9ASSET", "NAASSET", "NTLNLS", "ORE",
    "NETINC", "INTINC", "EINTEXP", "NONII", "NONIX", "ELNATR",
    "ROA", "ROE", "NIMY", "RBC1AAJ", "RBCRWAJ",
)

# Fields that cannot be negative under standard Call Report accounting rules
NON_NEGATIVE_COLS = (
    "ASSET", "DEP", "DEPDOM", "BRO", "CHBAL", "SC",
    "LNLSGR", "LNLSNET", "LNRE", "LNCI", "LNCON", "LNATRES",
    "NCLNLS", "P3ASSET", "P9ASSET", "NAASSET", "ORE",
    "INTINC", "EINTEXP", "NONIX",
)

# Fields that can legitimately be negative (losses, net charge-offs/recoveries, insolvency)
VALID_NEGATIVE_COLS = (
    "EQ", "NTLNLS", "NETINC", "NONII", "ELNATR",
    "ROA", "ROE", "NIMY", "RBC1AAJ", "RBCRWAJ",
)

RATIO_COLS = ("ROA", "ROE", "NIMY", "RBC1AAJ", "RBCRWAJ")

POLARS_SCHEMA = {
    "CERT": pl.Int64,
    "REPDTE": pl.String,
    "STNAME": pl.String,
    **{c: pl.Float64 for c in NUMERIC_COLS},
}
SCAN_CAST_OPTS = pl.ScanCastOptions(integer_cast="allow-float")


def load_raw_tables() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    fin = pl.scan_parquet(
        str(FIN_DIR / "*.parquet"), schema=POLARS_SCHEMA, cast_options=SCAN_CAST_OPTS
    ).collect()
    inst = pl.read_parquet(RAW_DIR / "institutions.parquet")
    fail = pl.read_parquet(RAW_DIR / "failures.parquet")
    return fin, inst, fail


def run_data_quality_audit(
    fin: pl.DataFrame, inst: pl.DataFrame, fail: pl.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate comprehensive column-level, decade-level, and exit-type audit tables."""
    n_total = fin.height
    dq_rows = []
    for col in NUMERIC_COLS:
        s = fin[col]
        n_null = int(s.null_count())
        n_zero = int((s == 0).sum())
        n_neg = int((s < 0).sum())

        # Structural zeros in capital ratios when equity is non-zero
        if col in ("RBCRWAJ", "RBC1AAJ"):
            n_struct_zero = int(((fin[col] == 0) & (fin["EQ"].is_not_null()) & (fin["EQ"] != 0)).sum())
        else:
            n_struct_zero = 0

        neg_category = (
            "Impossible (accounting error)"
            if col in NON_NEGATIVE_COLS
            else "Valid (loss / insolvency / recovery)"
        )

        dq_rows.append(
            {
                "column": col,
                "total_rows": n_total,
                "null_count": n_null,
                "null_pct": round(n_null / n_total * 100, 3),
                "zero_count": n_zero,
                "zero_pct": round(n_zero / n_total * 100, 3),
                "structural_zero_count": n_struct_zero,
                "structural_zero_pct": round(n_struct_zero / n_total * 100, 3),
                "neg_count": n_neg,
                "neg_pct": round(n_neg / n_total * 100, 4),
                "negative_interpretation": neg_category if n_neg > 0 else "None observed",
                "min": float(s.min()) if n_null < n_total else None,
                "p01": float(s.quantile(0.01)) if n_null < n_total else None,
                "p50": float(s.quantile(0.50)) if n_null < n_total else None,
                "p99": float(s.quantile(0.99)) if n_null < n_total else None,
                "max": float(s.max()) if n_null < n_total else None,
            }
        )
    dq_df = pd.DataFrame(dq_rows)
    dq_df.to_csv(OUT_DIR / "part2_dq_report.csv", index=False)

    # Decade-level missingness & structural zero audit
    fin_dec = fin.with_columns(
        ((pl.col("REPDTE").str.slice(0, 4).cast(pl.Int32) // 10) * 10).cast(pl.String).add("s").alias("decade")
    )
    decade_df = (
        fin_dec.group_by("decade")
        .agg(
            [
                pl.len().alias("rows"),
                pl.col("CERT").n_unique().alias("distinct_banks"),
                ((pl.col("RBCRWAJ") == 0) & (pl.col("EQ") != 0)).mean().mul(100).round(2).alias("rbcrwaj_struct_zero_pct"),
                ((pl.col("RBC1AAJ") == 0) & (pl.col("EQ") != 0)).mean().mul(100).round(2).alias("rbc1aaj_struct_zero_pct"),
                pl.col("P3ASSET").is_null().mean().mul(100).round(2).alias("p3asset_null_pct"),
                pl.col("P9ASSET").is_null().mean().mul(100).round(2).alias("p9asset_null_pct"),
                pl.col("NAASSET").is_null().mean().mul(100).round(2).alias("naasset_null_pct"),
                pl.col("NTLNLS").is_null().mean().mul(100).round(2).alias("ntlnls_null_pct"),
                pl.col("ELNATR").is_null().mean().mul(100).round(2).alias("elnatr_null_pct"),
                pl.col("BRO").is_null().mean().mul(100).round(2).alias("bro_null_pct"),
                pl.col("ROE").is_null().mean().mul(100).round(2).alias("roe_null_pct"),
            ]
        )
        .sort("decade")
        .to_pandas()
    )
    decade_df.to_csv(OUT_DIR / "part2_dq_by_decade.csv", index=False)

    # Accounting identity & reporting gap audit
    q_list = sorted(fin["REPDTE"].unique().to_list())
    q_map = pl.DataFrame({"REPDTE": q_list, "q_idx": list(range(len(q_list)))})
    fin_q = (
        fin.select(["CERT", "REPDTE", "ASSET", "DEP", "DEPDOM", "LNLSGR", "LNLSNET", "NCLNLS", "EQ"])
        .join(q_map, on="REPDTE")
        .sort(["CERT", "q_idx"])
        .with_columns((pl.col("q_idx") - pl.col("q_idx").shift(1).over("CERT")).alias("q_step"))
    )
    n_gaps = int(fin_q.filter(pl.col("q_step") > 1).height)
    n_gap_certs = int(fin_q.filter(pl.col("q_step") > 1)["CERT"].n_unique())

    # Bank exit classification across all 24,064 panel CERTs
    fail_post84 = (
        fail.filter(pl.col("CERT").is_not_null())
        .with_columns(
            pl.col("CERT").cast(pl.Int64),
            pl.col("FAILDATE").str.to_date("%m/%d/%Y").alias("fail_dt"),
        )
        .filter(pl.col("fail_dt") >= pl.date(1984, 1, 1))
    )
    fail_certs = fail_post84.group_by("CERT").agg(
        [
            pl.col("fail_dt").min().alias("first_fail_dt"),
            pl.col("fail_dt").max().alias("last_fail_dt"),
            pl.col("RESTYPE").first().alias("restype"),
        ]
    )

    bank_span = (
        fin.group_by("CERT")
        .agg(
            [
                pl.col("REPDTE").min().alias("first_repdte"),
                pl.col("REPDTE").max().alias("last_repdte"),
                pl.len().alias("n_quarters"),
            ]
        )
        .join(inst.select(["CERT", "ACTIVE", "CHANGEC1", "BKCLASS"]), on="CERT", how="left")
        .join(fail_certs, on="CERT", how="left")
        .with_columns(
            pl.when(pl.col("first_fail_dt").is_not_null())
            .then(pl.lit("Failed / Assisted (in failures.parquet >=1984)"))
            .when((pl.col("ACTIVE") == 1) | (pl.col("last_repdte") == "20260630"))
            .then(pl.lit("Active at end of sample (2026Q2)"))
            .when(pl.col("CHANGEC1").is_not_null())
            .then(pl.lit("Merger / Voluntary Exit (CHANGEC1 non-failure)"))
            .otherwise(pl.lit("Historical Pre-FIRREA Thrift / Unmatched Exit"))
            .alias("exit_category")
        )
    )
    exit_df = (
        bank_span.group_by("exit_category")
        .agg(
            [
                pl.len().alias("distinct_banks"),
                pl.col("n_quarters").sum().alias("total_bank_quarters"),
                pl.col("n_quarters").mean().round(1).alias("avg_quarters_reported"),
            ]
        )
        .sort("distinct_banks", descending=True)
        .to_pandas()
    )
    exit_df.to_csv(OUT_DIR / "part2_exit_types.csv", index=False)

    # Save anomaly summary counts
    anomaly_summary = pd.DataFrame(
        [
            {"check": "Total raw bank-quarter rows", "count": n_total},
            {"check": "Distinct banks (CERT)", "count": int(fin["CERT"].n_unique())},
            {"check": "Discontinuous reporting gaps (q_step > 1 quarter)", "count": n_gaps},
            {"check": "Distinct banks with reporting gaps", "count": n_gap_certs},
            {"check": "Rows with EQ < 0 (technical book insolvency)", "count": int((fin["EQ"] < 0).sum())},
            {"check": "Rows with DEPDOM > DEP", "count": int((fin["DEPDOM"] > fin["DEP"]).sum())},
            {"check": "Rows with LNLSNET > LNLSGR", "count": int((fin["LNLSNET"] > fin["LNLSGR"]).sum())},
            {"check": "Rows with impossible negative stocks/flows (any column)", "count": int(
                fin.select([pl.col(c) < 0 for c in NON_NEGATIVE_COLS]).sum_horizontal().gt(0).sum()
            )},
        ]
    )
    anomaly_summary.to_csv(OUT_DIR / "part2_anomaly_checks.csv", index=False)
    return dq_df, decade_df, exit_df


def build_clean_labeled_panel(
    fin: pl.DataFrame, inst: pl.DataFrame, fail: pl.DataFrame
) -> tuple[pl.DataFrame, pd.DataFrame]:
    """
    Apply strict point-in-time cleaning rules and attach forward 4-quarter failure labels.
    """
    # 1. Parse post-1984 failure events
    fail_post84 = (
        fail.filter(pl.col("CERT").is_not_null())
        .with_columns(
            pl.col("CERT").cast(pl.Int64),
            pl.col("FAILDATE").str.to_date("%m/%d/%Y").alias("fail_dt"),
        )
        .filter(pl.col("fail_dt") >= pl.date(1984, 1, 1))
    )

    # Quarter index mapping across all 170 quarters
    q_list = sorted(fin["REPDTE"].unique().to_list())
    q_map = pl.DataFrame({"REPDTE": q_list, "q_idx": list(range(len(q_list)))})

    panel = (
        fin.join(q_map, on="REPDTE", how="left")
        .with_columns(
            pl.col("REPDTE").str.to_date("%Y%m%d").alias("rep_dt"),
            pl.col("REPDTE").str.to_date("%Y%m%d").dt.offset_by("1y").alias("rep_dt_plus_4q"),
        )
    )

    # 2. Match failure events to each (CERT, REPDTE) to identify:
    #    (a) post-failure conservatorship rows (rep_dt >= fail_dt for RESTYPE == 'FAILURE')
    #    (b) future failures within (rep_dt, rep_dt + 4 quarters]
    fail_join = (
        panel.select(["CERT", "REPDTE", "rep_dt", "rep_dt_plus_4q"])
        .join(
            fail_post84.select(["CERT", "fail_dt", "RESTYPE", "COST", "QBFASSET", "QBFDEP"]),
            on="CERT",
            how="left",
        )
        .with_columns(
            ((pl.col("fail_dt") > pl.col("rep_dt")) & (pl.col("fail_dt") <= pl.col("rep_dt_plus_4q"))).alias("is_fail_4q"),
            ((pl.col("fail_dt") <= pl.col("rep_dt")) & (pl.col("RESTYPE") == "FAILURE")).alias("is_post_fail"),
        )
        .group_by(["CERT", "REPDTE"])
        .agg(
            [
                pl.col("is_fail_4q").any().cast(pl.Int8).alias("y_fail_4q"),
                pl.col("is_post_fail").any().alias("already_failed_outright"),
                pl.col("fail_dt").filter(pl.col("is_fail_4q")).min().alias("target_fail_dt"),
                pl.col("COST").filter(pl.col("is_fail_4q")).first().alias("target_fail_cost"),
                pl.col("QBFASSET").filter(pl.col("is_fail_4q")).first().alias("target_fail_qbfasset"),
                pl.col("RESTYPE").filter(pl.col("is_fail_4q")).first().alias("target_fail_restype"),
            ]
        )
    )

    panel = panel.join(fail_join, on=["CERT", "REPDTE"], how="left")

    # Exclude post-failure conservatorship rows (1,095 rows where bank already failed outright)
    n_before_drop = panel.height
    panel = panel.filter(~pl.col("already_failed_outright")).drop("already_failed_outright")
    n_dropped_post_fail = n_before_drop - panel.height
    print(f"Excluded {n_dropped_post_fail:,} post-failure conservatorship Call Report rows.")

    # 3. Point-in-time structural zero & impossible negative cleaning
    #    - Recode RBCRWAJ == 0 (when EQ != 0) and RBC1AAJ == 0 (when EQ != 0) to null
    #    - Record binary missingness/validity indicator flags at row (i, q)
    impossible_neg_exprs = [pl.col(c) < 0 for c in NON_NEGATIVE_COLS]
    panel = panel.with_columns(
        [
            pl.col("P3ASSET").is_null().cast(pl.Int8).alias("flag_p3asset_missing"),
            ((pl.col("RBCRWAJ") == 0) & (pl.col("EQ").is_not_null()) & (pl.col("EQ") != 0))
            .cast(pl.Int8)
            .alias("flag_rbcrwaj_struct_zero"),
            ((pl.col("RBC1AAJ") == 0) & (pl.col("EQ").is_not_null()) & (pl.col("EQ") != 0))
            .cast(pl.Int8)
            .alias("flag_rbc1aaj_struct_zero"),
            pl.sum_horizontal(impossible_neg_exprs).gt(0).cast(pl.Int8).alias("flag_impossible_neg"),
            (pl.col("EQ") < 0).fill_null(False).cast(pl.Int8).alias("flag_negative_equity"),
        ]
    ).with_columns(
        [
            pl.when((pl.col("RBCRWAJ") == 0) & (pl.col("EQ").is_not_null()) & (pl.col("EQ") != 0))
            .then(None)
            .otherwise(pl.col("RBCRWAJ"))
            .alias("RBCRWAJ"),
            pl.when((pl.col("RBC1AAJ") == 0) & (pl.col("EQ").is_not_null()) & (pl.col("EQ") != 0))
            .then(None)
            .otherwise(pl.col("RBC1AAJ"))
            .alias("RBC1AAJ"),
        ]
        + [
            pl.when(pl.col(c) < 0).then(None).otherwise(pl.col(c)).alias(c)
            for c in NON_NEGATIVE_COLS
        ]
    )

    # 4. Causal 1-quarter forward-fill strictly within contiguous reporting spells
    panel = panel.sort(["CERT", "q_idx"]).with_columns(
        (pl.col("q_idx") - pl.col("q_idx").shift(1).over("CERT")).alias("q_step")
    ).with_columns(
        (pl.col("q_step") > 1).fill_null(False).cum_sum().cast(pl.Int64).over("CERT").alias("spell_id")
    ).with_columns(
        [
            pl.col(c).forward_fill(limit=1).over(["CERT", "spell_id"]).alias(c)
            for c in NUMERIC_COLS
        ]
    )

    # 5. Join static institution metadata (BKCLASS, ACTIVE, CHANGEC1)
    inst_dedup = inst.unique(subset=["CERT"], keep="first").select(
        ["CERT", "BKCLASS", "ACTIVE", "CHANGEC1"]
    )
    panel = panel.join(inst_dedup, on="CERT", how="left")

    # 6. Assign temporal split and restrict modeling panel to REPDTE <= 20250630
    panel = panel.with_columns(
        pl.when(pl.col("REPDTE") <= "20061231")
        .then(pl.lit("dev"))
        .when((pl.col("REPDTE") >= "20070331") & (pl.col("REPDTE") <= "20071231"))
        .then(pl.lit("embargo"))
        .when((pl.col("REPDTE") >= "20080331") & (pl.col("REPDTE") <= "20250630"))
        .then(pl.lit("test"))
        .otherwise(pl.lit("right_censored"))
        .alias("split")
    )

    # Summarize all 4 temporal regimes before filtering out right_censored
    split_rows = []
    for sp in ("dev", "embargo", "test", "right_censored"):
        sub = panel.filter(pl.col("split") == sp)
        pos = sub.filter(pl.col("y_fail_4q") == 1)
        split_rows.append(
            {
                "split": sp,
                "start_repdte": sub["REPDTE"].min(),
                "end_repdte": sub["REPDTE"].max(),
                "n_quarters": int(sub["REPDTE"].n_unique()),
                "bank_quarters": sub.height,
                "distinct_banks": int(sub["CERT"].n_unique()),
                "positive_labels_y1": pos.height,
                "positive_rate_pct": round(pos.height / sub.height * 100, 4),
                "distinct_failing_certs": int(pos["CERT"].n_unique()),
            }
        )
    split_df = pd.DataFrame(split_rows)
    split_df.to_csv(OUT_DIR / "part2_split_summary.csv", index=False)

    # 7. Point-in-time ratio winsorization (0.1% / 99.9%) fitted strictly on Development split (<= 2006Q4)
    dev_sub = panel.filter(pl.col("split") == "dev")
    bounds_rows = []
    win_exprs = []
    for rc in RATIO_COLS:
        lo = float(dev_sub[rc].quantile(0.001))
        hi = float(dev_sub[rc].quantile(0.999))
        bounds_rows.append({"column": rc, "dev_p001": round(lo, 4), "dev_p999": round(hi, 4)})
        win_exprs.append(pl.col(rc).clip(lo, hi).alias(f"{rc}_win"))
    pd.DataFrame(bounds_rows).to_csv(OUT_DIR / "part2_dev_winsor_bounds.csv", index=False)
    panel = panel.with_columns(win_exprs)

    # Keep only quarters <= 2025Q2 (dev, embargo, test) for the labeled modeling panel
    panel_labeled = panel.filter(pl.col("REPDTE") <= "20250630").sort(["CERT", "REPDTE"])
    out_path = CLEAN_DIR / "panel_labeled.parquet"
    panel_labeled.write_parquet(out_path, compression="snappy")
    print(f"Saved clean labeled panel ({panel_labeled.height:,} rows, {len(panel_labeled.columns)} cols) to {out_path}")
    return panel_labeled, split_df


def plot_data_quality_overview(decade_df: pd.DataFrame, panel_labeled: pl.DataFrame) -> None:
    """Plot structural missingness by decade and quarterly failure rate across splits."""
    q_agg = (
        panel_labeled.group_by(["REPDTE", "rep_dt", "split"])
        .agg(
            [
                pl.len().alias("n_banks"),
                pl.col("y_fail_4q").sum().alias("n_fail_4q"),
                (pl.col("y_fail_4q").mean() * 100).alias("fail_rate_pct"),
            ]
        )
        .sort("rep_dt")
        .to_pandas()
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.5), dpi=200)

    # Panel A: Structural missingness & zeros by decade
    x = range(len(decade_df))
    w = 0.25
    axes[0].bar(
        [i - w for i in x],
        decade_df["rbcrwaj_struct_zero_pct"],
        width=w,
        label="RBCRWAJ Structural Zero (Pre-Basel I / CBLR)",
        color="#1f77b4",
    )
    axes[0].bar(
        x,
        decade_df["p3asset_null_pct"],
        width=w,
        label="P3ASSET Null (Confidential pre-2001)",
        color="#ff7f0e",
    )
    axes[0].bar(
        [i + w for i in x],
        decade_df["p9asset_null_pct"],
        width=w,
        label="P9ASSET / NAASSET Null (1980s TFR Thrifts)",
        color="#2ca02c",
    )
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(decade_df["decade"])
    axes[0].set_ylabel("% of Bank-Quarters")
    axes[0].set_title("A. Structural Missingness & Sentinel Zeros by Decade")
    axes[0].legend(fontsize=8, loc="upper right")

    # Panel B: Quarterly 4-quarter ahead failure rate & temporal split boundaries
    axes[1].plot(q_agg["rep_dt"], q_agg["fail_rate_pct"], color="#d62728", linewidth=1.6, label="4Q Forward Failure Rate (%)")
    axes[1].axvline(pd.to_datetime("2006-12-31"), color="#1f77b4", linestyle="--", linewidth=1.3, label="Dev End (2006Q4)")
    axes[1].axvline(pd.to_datetime("2008-03-31"), color="#2ca02c", linestyle="--", linewidth=1.3, label="Test Start (2008Q1)")
    axes[1].axvspan(pd.to_datetime("2007-01-01"), pd.to_datetime("2008-03-31"), color="gray", alpha=0.2, label="4Q Embargo (2007)")
    axes[1].set_ylabel("Failure Rate within Next 4 Quarters (%)")
    axes[1].set_xlabel("Report Date (REPDTE)")
    axes[1].set_title("B. Forward 4Q Failure Rate & Leak-Free Temporal Split")
    axes[1].legend(fontsize=8, loc="upper right")

    fig.suptitle("Part 2: FDIC Call Report Data Quality Audit & Temporal Split Design", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "part2_dq_overview.png", dpi=200)
    plt.close(fig)


def main() -> None:
    print("=== Part 2: Loading Raw FDIC Tables ===")
    fin, inst, fail = load_raw_tables()
    print(f"Loaded financials: {fin.height:,} rows | institutions: {inst.height:,} | failures: {fail.height:,}")

    print("\n=== Part 2: Running Automated Data Quality Audit ===")
    dq_df, decade_df, exit_df = run_data_quality_audit(fin, inst, fail)
    print("\n--- Decade Missingness & Structural Zero Summary ---")
    print(decade_df.to_string(index=False))
    print("\n--- Bank Exit Classification (24,064 Distinct Banks) ---")
    print(exit_df.to_string(index=False))

    print("\n=== Part 2: Building Point-in-Time Clean Labeled Panel ===")
    panel_labeled, split_df = build_clean_labeled_panel(fin, inst, fail)
    print("\n--- Temporal Split & Failure Label Summary ---")
    print(split_df.to_string(index=False))

    plot_data_quality_overview(decade_df, panel_labeled)
    print(f"\nSaved Part 2 reports and charts to {OUT_DIR}")


if __name__ == "__main__":
    main()
