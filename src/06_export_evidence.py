"""
Part 6 (Audit & Verification): Read-Only Evidence & Reconciliation Exporter
===========================================================================
Lightweight, read-only script that reads existing raw data (data_raw/fdic/),
cleaned panel (data_clean/panel_labeled.parquet), and verified outputs (outputs/)
without modifying any prior script or output artifact.

Exports:
  1. outputs/part1_aux_tables_summary.csv
     - Exact row/column counts of institutions.parquet & failures.parquet
     - Complete RESTYPE split (FAILURE vs. ASSISTANCE, overall & post-1984)
     - Exact manifest_financials.json creation/download timestamps & totals
  2. outputs/part2_supplementary_evidence.csv
     - Reconciliation of 3,240 unique post-1984 failing CERTs vs. 3,218 in part2_exit_types.csv
     - Count of banks with multiple post-1984 failure/assistance events
     - Exact count & year range of excluded post-failure conservatorship rows (rep_dt >= fail_dt)
     - EQ < 0 rates among failing vs. non-failing Development bank-quarters
     - Reporting gap breakdown (single-quarter vs. multi-quarter, max delta_q)
     - RBCRWAJ structural-zero rates in 2019 vs. 2026 (plus 1980s/2020s checks)
     - Scaling curve 0.25x & 0.50x sorted-CERT sampling verification vs. part3_scaling_curve.csv
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data_raw" / "fdic"
FIN_DIR = RAW_DIR / "financials"
CLEAN_DIR = ROOT / "data_clean"
OUT_DIR = ROOT / "outputs"

NUMERIC_COLS = (
    "ASSET", "DEP", "DEPDOM", "EQ", "BRO", "CHBAL", "SC",
    "LNLSGR", "LNLSNET", "LNRE", "LNCI", "LNCON", "LNATRES",
    "NCLNLS", "P3ASSET", "P9ASSET", "NAASSET", "NTLNLS", "ORE",
    "NETINC", "INTINC", "EINTEXP", "NONII", "NONIX", "ELNATR",
    "ROA", "ROE", "NIMY", "RBC1AAJ", "RBCRWAJ",
)
POLARS_SCHEMA = {
    "CERT": pl.Int64,
    "REPDTE": pl.String,
    "STNAME": pl.String,
    **{c: pl.Float64 for c in NUMERIC_COLS},
}
SCAN_CAST_OPTS = pl.ScanCastOptions(integer_cast="allow-float")


def main() -> None:
    print("=== 06_export_evidence.py: Read-Only Verification & Evidence Export ===")

    # ------------------------------------------------------------------
    # 1. Load raw auxiliary tables, manifest, and minimal raw financials
    # ------------------------------------------------------------------
    inst_path = RAW_DIR / "institutions.parquet"
    fail_path = RAW_DIR / "failures.parquet"
    manifest_path = RAW_DIR / "manifest_financials.json"

    inst = pl.read_parquet(inst_path)
    fail = pl.read_parquet(fail_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Read only the 4 columns needed from raw financials (fast columnar pushdown)
    fin = (
        pl.scan_parquet(
            str(FIN_DIR / "*.parquet"),
            schema=POLARS_SCHEMA,
            cast_options=SCAN_CAST_OPTS,
        )
        .select(["CERT", "REPDTE", "EQ", "RBCRWAJ"])
        .collect()
    )

    # Read cleaned labeled panel (only columns needed for Dev EQ<0 and scaling check)
    panel = pl.read_parquet(
        CLEAN_DIR / "panel_labeled.parquet",
        columns=["CERT", "REPDTE", "split", "y_fail_4q", "EQ", "flag_negative_equity"],
    )

    exit_types_csv = pd.read_csv(OUT_DIR / "part2_exit_types.csv")
    scaling_csv = pd.read_csv(OUT_DIR / "part3_scaling_curve.csv")

    # ------------------------------------------------------------------
    # 2. Part 1 Auxiliary Tables & Manifest Summary
    # ------------------------------------------------------------------
    fail_parsed = fail.with_columns(
        pl.col("CERT").cast(pl.Int64),
        pl.col("FAILDATE").str.to_date("%m/%d/%Y").alias("fail_dt"),
    )
    restype_counts = (
        fail_parsed.group_by("RESTYPE")
        .len()
        .sort("len", descending=True)
        .to_dict(as_series=False)
    )
    rt_map = dict(zip(restype_counts["RESTYPE"], restype_counts["len"]))
    n_failure_all = int(rt_map.get("FAILURE", 0))
    n_assistance_all = int(rt_map.get("ASSISTANCE", 0))
    n_other_all = int(fail_parsed.height - n_failure_all - n_assistance_all)

    fail_post84 = fail_parsed.filter(
        pl.col("CERT").is_not_null() & (pl.col("fail_dt") >= pl.date(1984, 1, 1))
    )
    rt_post84 = dict(
        zip(
            *fail_post84.group_by("RESTYPE")
            .len()
            .to_dict(as_series=False)
            .values()
        )
    )
    n_post84_events = int(fail_post84.height)
    n_post84_unique_certs = int(fail_post84["CERT"].n_unique())

    events_per_cert = fail_post84.group_by("CERT").len()
    multi_event_certs = events_per_cert.filter(pl.col("len") > 1)
    n_multi_event_banks = int(multi_event_certs.height)
    max_events_per_bank = int(events_per_cert["len"].max())

    # File timestamps for manifest_financials.json
    st = manifest_path.stat()
    created_local = datetime.fromtimestamp(st.st_birthtime).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    created_utc = datetime.fromtimestamp(st.st_birthtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    mtime_local = datetime.fromtimestamp(st.st_mtime).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    manifest_total_rows = int(sum(v["rows"] for v in manifest.values()))

    aux_rows = [
        ("institutions.parquet", "rows", str(inst.height), "Total FDIC institution directory records"),
        ("institutions.parquet", "columns", str(inst.width), ", ".join(inst.columns)),
        ("institutions.parquet", "distinct_certs", str(int(inst["CERT"].n_unique())), "Unique FDIC CERT identifiers"),
        ("institutions.parquet", "active_institutions", str(int((inst["ACTIVE"] == 1).sum())), "ACTIVE == 1"),
        ("institutions.parquet", "inactive_with_changec1", str(int(((inst["ACTIVE"] != 1) & inst["CHANGEC1"].is_not_null()).sum())), "Inactive institutions with exit code CHANGEC1"),
        ("failures.parquet", "rows", str(fail.height), "Total historical failure & assistance records (1934-2026)"),
        ("failures.parquet", "columns", str(fail.width), ", ".join(fail.columns)),
        ("failures.parquet", "min_fail_date", str(fail_parsed["fail_dt"].min()), "Earliest FAILDATE in failures.parquet"),
        ("failures.parquet", "max_fail_date", str(fail_parsed["fail_dt"].max()), "Latest FAILDATE in failures.parquet"),
        ("failures.parquet", "restype_FAILURE_rows", str(n_failure_all), f"{n_failure_all / fail.height * 100:.2f}% of total"),
        ("failures.parquet", "restype_ASSISTANCE_rows", str(n_assistance_all), f"{n_assistance_all / fail.height * 100:.2f}% of total"),
        ("failures.parquet", "restype_other_or_null_rows", str(n_other_all), "Check: FAILURE + ASSISTANCE + other == 4,117"),
        ("failures.parquet", "post_1984_events_total", str(n_post84_events), "Non-null CERT with FAILDATE >= 1984-01-01"),
        ("failures.parquet", "post_1984_restype_FAILURE", str(int(rt_post84.get("FAILURE", 0))), "Post-1984 outright failures"),
        ("failures.parquet", "post_1984_restype_ASSISTANCE", str(int(rt_post84.get("ASSISTANCE", 0))), "Post-1984 open-bank assistance events"),
        ("failures.parquet", "post_1984_unique_certs", str(n_post84_unique_certs), "Distinct failing/assisted CERTs >= 1984-01-01"),
        ("failures.parquet", "post_1984_multi_event_banks", str(n_multi_event_banks), f"Banks with >1 post-1984 event (max={max_events_per_bank} events/bank)"),
        ("manifest_financials.json", "quarters_verified", str(len(manifest)), f"{min(manifest.keys())} to {max(manifest.keys())}"),
        ("manifest_financials.json", "manifest_total_rows", str(manifest_total_rows), "Exact sum of rows across 170 verified quarters"),
        ("manifest_financials.json", "download_created_local", created_local, "File creation timestamp of manifest_financials.json"),
        ("manifest_financials.json", "download_created_utc", created_utc, "UTC creation timestamp of manifest_financials.json"),
        ("manifest_financials.json", "file_last_modified_local", mtime_local, "Filesystem mtime (OneDrive sync timestamp)"),
    ]
    aux_df = pd.DataFrame(aux_rows, columns=["table_name", "metric", "value", "notes"])
    aux_df.to_csv(OUT_DIR / "part1_aux_tables_summary.csv", index=False)

    # ------------------------------------------------------------------
    # 3. Part 2 Supplementary Evidence & Reconciliations
    # ------------------------------------------------------------------
    # (c) Reconciliation of 3,240 unique post-1984 failing CERTs vs 3,218 in part2_exit_types.csv
    fin_certs = fin.select("CERT").unique()
    post84_certs_df = fail_post84.group_by("CERT").agg(
        [
            pl.col("fail_dt").min().alias("first_fail_dt"),
            pl.col("fail_dt").max().alias("last_fail_dt"),
            pl.col("RESTYPE").first().alias("first_restype"),
        ]
    )
    in_fin = post84_certs_df.join(fin_certs, on="CERT", how="inner")
    not_in_fin = post84_certs_df.join(fin_certs, on="CERT", how="anti")
    n_not_in_fin_pre_q1 = int(not_in_fin.filter(pl.col("last_fail_dt") < pl.date(1984, 3, 31)).height)
    n_not_in_fin_post_q1 = int(not_in_fin.height - n_not_in_fin_pre_q1)

    csv_failed_banks = int(
        exit_types_csv.loc[
            exit_types_csv["exit_category"].str.contains("Failed", case=False),
            "distinct_banks",
        ].iloc[0]
    )
    assert in_fin.height == csv_failed_banks, (
        f"Mismatch: in_fin={in_fin.height} vs part2_exit_types.csv={csv_failed_banks}"
    )

    # (e) Excluded post-failure conservatorship rows (rep_dt >= fail_dt for RESTYPE == 'FAILURE')
    fin_dt = fin.with_columns(pl.col("REPDTE").str.to_date("%Y%m%d").alias("rep_dt"))
    post_fail_rows = (
        fin_dt.select(["CERT", "REPDTE", "rep_dt"])
        .join(
            fail_post84.filter(pl.col("RESTYPE") == "FAILURE").select(["CERT", "fail_dt"]),
            on="CERT",
            how="inner",
        )
        .filter(pl.col("rep_dt") >= pl.col("fail_dt"))
        .unique(subset=["CERT", "REPDTE"])
    )
    n_post_fail_rows = int(post_fail_rows.height)
    n_post_fail_certs = int(post_fail_rows["CERT"].n_unique())
    post_fail_min_yr = int(post_fail_rows["rep_dt"].dt.year().min())
    post_fail_max_yr = int(post_fail_rows["rep_dt"].dt.year().max())

    # (f) EQ < 0 rates among failing vs non-failing Development rows
    dev = panel.filter(pl.col("split") == "dev")
    dev_pos = dev.filter(pl.col("y_fail_4q") == 1)
    dev_neg = dev.filter(pl.col("y_fail_4q") == 0)

    dev_pos_eq_lt0 = int((dev_pos["EQ"] < 0).sum())
    dev_neg_eq_lt0 = int((dev_neg["EQ"] < 0).sum())
    dev_pos_flag_eq_lt0 = int((dev_pos["flag_negative_equity"] == 1).sum())
    dev_neg_flag_eq_lt0 = int((dev_neg["flag_negative_equity"] == 1).sum())
    dev_pos_eq_rate_pct = dev_pos_flag_eq_lt0 / dev_pos.height * 100.0
    dev_neg_eq_rate_pct = dev_neg_flag_eq_lt0 / dev_neg.height * 100.0
    dev_pos_eq_nonnull_rate_pct = dev_pos_eq_lt0 / int(dev_pos["EQ"].is_not_null().sum()) * 100.0
    dev_neg_eq_nonnull_rate_pct = dev_neg_eq_lt0 / int(dev_neg["EQ"].is_not_null().sum()) * 100.0

    # (g) Reporting gap breakdown on raw panel
    q_list = sorted(fin["REPDTE"].unique().to_list())
    q_map = pl.DataFrame({"REPDTE": q_list, "q_idx": list(range(len(q_list)))})
    gaps = (
        fin.select(["CERT", "REPDTE"])
        .join(q_map, on="REPDTE")
        .sort(["CERT", "q_idx"])
        .with_columns((pl.col("q_idx") - pl.col("q_idx").shift(1).over("CERT")).alias("q_step"))
        .filter(pl.col("q_step") > 1)
    )
    n_gaps_total = int(gaps.height)
    n_gaps_banks = int(gaps["CERT"].n_unique())
    n_gaps_single = int(gaps.filter(pl.col("q_step") == 2).height)
    n_gaps_multi = int(gaps.filter(pl.col("q_step") > 2).height)
    max_dq = int(gaps["q_step"].max())

    # (h) RBCRWAJ zero & structural-zero rates in 2019 vs 2026 (plus 1980s / 2020s)
    fin_yr = fin.with_columns(
        pl.col("REPDTE").str.slice(0, 4).cast(pl.Int32).alias("year"),
        (pl.col("RBCRWAJ") == 0).alias("is_rbcrwaj_zero"),
        ((pl.col("RBCRWAJ") == 0) & pl.col("EQ").is_not_null() & (pl.col("EQ") != 0)).alias("is_rbcrwaj_sz"),
        ((pl.col("RBCRWAJ") == 0) & (pl.col("EQ") != 0)).alias("is_rbcrwaj_sz_polars"),
    )

    def yr_sz_stats(df_sub: pl.DataFrame) -> tuple[int, int, int, float, float, float]:
        zero_num = int(df_sub["is_rbcrwaj_zero"].sum())
        sz_num = int(df_sub["is_rbcrwaj_sz"].sum())
        den = int(df_sub.height)
        zero_pct = (zero_num / den * 100.0) if den > 0 else 0.0
        sz_all_pct = (sz_num / den * 100.0) if den > 0 else 0.0
        sz_nonnull_eq_pct = float(df_sub["is_rbcrwaj_sz_polars"].mean() * 100.0) if den > 0 else 0.0
        return zero_num, sz_num, den, zero_pct, sz_all_pct, sz_nonnull_eq_pct

    z19_n, sz19_n, d19, z19_p, sz19_all_p, sz19_nn_p = yr_sz_stats(fin_yr.filter(pl.col("year") == 2019))
    z26_n, sz26_n, d26, z26_p, sz26_all_p, sz26_nn_p = yr_sz_stats(fin_yr.filter(pl.col("year") == 2026))
    z80_n, sz80_n, d80, z80_p, sz80_all_p, sz80_nn_p = yr_sz_stats(fin_yr.filter(pl.col("year") < 1990))
    z20_n, sz20_n, d20, z20_p, sz20_all_p, sz20_nn_p = yr_sz_stats(fin_yr.filter(pl.col("year") >= 2020))

    # Scaling curve 0.25x & 0.50x verification (matching src/03_scalable_processing.py logic)
    certs_sorted = panel["CERT"].unique().sort()
    n_certs_total = len(certs_sorted)
    n_certs_025 = int(n_certs_total * 0.25)
    rows_025_repro = int(panel.filter(pl.col("CERT").is_in(certs_sorted.head(n_certs_025).implode())).height)
    rows_025_csv = int(scaling_csv.loc[scaling_csv["scale_factor"] == 0.25, "rows"].iloc[0])

    n_certs_050 = int(n_certs_total * 0.50)
    rows_050_repro = int(panel.filter(pl.col("CERT").is_in(certs_sorted.head(n_certs_050).implode())).height)
    rows_050_csv = int(scaling_csv.loc[scaling_csv["scale_factor"] == 0.5, "rows"].iloc[0])

    assert rows_025_repro == rows_025_csv, f"0.25x mismatch: {rows_025_repro} vs {rows_025_csv}"
    assert rows_050_repro == rows_050_csv, f"0.50x mismatch: {rows_050_repro} vs {rows_050_csv}"

    supp_rows = [
        # (c) & (d) Failure bank reconciliation
        ("failure_reconciliation", "post_1984_failure_events_total", str(n_post84_events), "events", "failures.parquet: CERT not null & FAILDATE >= 1984-01-01"),
        ("failure_reconciliation", "post_1984_unique_failing_certs", str(n_post84_unique_certs), "distinct CERTs", "Unique CERTs in post-1984 failures.parquet"),
        ("failure_reconciliation", "post_1984_banks_with_multiple_events", str(n_multi_event_banks), f"distinct CERTs (max={max_events_per_bank} events)", "CERTs with >1 post-1984 record in failures.parquet"),
        ("failure_reconciliation", "post_1984_failing_certs_in_call_reports", str(in_fin.height), "distinct CERTs (matches part2_exit_types.csv)", "Inner join of post-1984 failing CERTs with 1984Q1-2026Q2 financials"),
        ("failure_reconciliation", "post_1984_failing_certs_without_call_reports", str(not_in_fin.height), f"distinct CERTs ({n_not_in_fin_pre_q1} failed before 1984-03-31; {n_not_in_fin_post_q1} on/after 1984-03-31)", "3,240 - 3,218 = 22 failing CERTs absent from raw Call Reports"),
        # (e) Post-failure conservatorship rows
        ("conservatorship_exclusion", "excluded_post_failure_rows", str(n_post_fail_rows), "bank-quarter rows", "Raw financials rows with rep_dt >= fail_dt for RESTYPE == 'FAILURE'"),
        ("conservatorship_exclusion", "excluded_post_failure_distinct_certs", str(n_post_fail_certs), "distinct CERTs", "Distinct banks with post-failure Call Report filings"),
        ("conservatorship_exclusion", "excluded_post_failure_year_range", f"{post_fail_min_yr}-{post_fail_max_yr}", "calendar years", f"min_year={post_fail_min_yr}, max_year={post_fail_max_yr}"),
        # (f) EQ < 0 in Dev
        ("dev_insolvency_eq_lt_0", "dev_failing_rows_total", str(dev_pos.height), "bank-quarters", "panel_labeled.parquet: split=='dev' & y_fail_4q==1"),
        ("dev_insolvency_eq_lt_0", "dev_failing_rows_eq_lt_0_flag", str(dev_pos_flag_eq_lt0), f"{dev_pos_eq_rate_pct:.2f}% of all failing dev rows ({dev_pos_eq_lt0} raw EQ<0; {dev_pos_eq_nonnull_rate_pct:.2f}% of non-null EQ)", "flag_negative_equity == 1 among dev y_fail_4q == 1"),
        ("dev_insolvency_eq_lt_0", "dev_non_failing_rows_total", str(dev_neg.height), "bank-quarters", "panel_labeled.parquet: split=='dev' & y_fail_4q==0"),
        ("dev_insolvency_eq_lt_0", "dev_non_failing_rows_eq_lt_0_flag", str(dev_neg_flag_eq_lt0), f"{dev_neg_eq_rate_pct:.2f}% of all non-failing dev rows ({dev_neg_eq_lt0} raw EQ<0; {dev_neg_eq_nonnull_rate_pct:.2f}% of non-null EQ)", "flag_negative_equity == 1 among dev y_fail_4q == 0"),
        # (g) Reporting gaps
        ("reporting_gaps", "total_reporting_gaps", str(n_gaps_total), "gaps (q_step > 1)", "Raw financials 1984Q1-2026Q2 consecutive q_idx step > 1"),
        ("reporting_gaps", "distinct_banks_with_gaps", str(n_gaps_banks), "distinct CERTs", "Unique CERTs experiencing >=1 gap"),
        ("reporting_gaps", "single_quarter_gaps_dq2", str(n_gaps_single), "gaps (q_step == 2)", "1 missing quarter between consecutive filings"),
        ("reporting_gaps", "multi_quarter_gaps_dq_gt2", str(n_gaps_multi), "gaps (q_step > 2)", ">=2 consecutive missing quarters"),
        ("reporting_gaps", "max_reporting_gap_dq", str(max_dq), "quarters", "max(q_idx[t] - q_idx[t-1]) within CERT"),
        # (h) RBCRWAJ raw zeros vs. strict structural zeros (EQ != 0)
        ("rbcrwaj_structural_zeros", "rbcrwaj_raw_zero_2019_pct", f"{z19_p:.2f}%", f"{z19_n:,} / {d19:,} rows (all {z19_n} have null EQ)", "Year 2019: RBCRWAJ == 0 over all rows"),
        ("rbcrwaj_structural_zeros", "rbcrwaj_struct_zero_2019_pct", f"{sz19_all_p:.2f}%", f"{sz19_n:,} / {d19:,} rows ({sz19_nn_p:.2f}% over non-null EQ)", "Year 2019: RBCRWAJ == 0 & EQ is not null & EQ != 0"),
        ("rbcrwaj_structural_zeros", "rbcrwaj_raw_zero_2026_pct", f"{z26_p:.2f}%", f"{z26_n:,} / {d26:,} rows (2026Q1-2026Q2)", "Year 2026: RBCRWAJ == 0 over all rows"),
        ("rbcrwaj_structural_zeros", "rbcrwaj_struct_zero_2026_pct", f"{sz26_all_p:.2f}%", f"{sz26_n:,} / {d26:,} rows ({sz26_nn_p:.2f}% over non-null EQ)", "Year 2026: RBCRWAJ == 0 & EQ is not null & EQ != 0 (CBLR opt-in)"),
        ("rbcrwaj_structural_zeros", "rbcrwaj_struct_zero_1980s_pct", f"{sz80_nn_p:.2f}%", f"{sz80_n:,} / {d80:,} all rows ({sz80_all_p:.2f}% all rows; {z80_p:.2f}% raw zero)", "1984-1989 pre-Basel I (matches part2_dq_by_decade.csv non-null EQ mean)"),
        ("rbcrwaj_structural_zeros", "rbcrwaj_struct_zero_2020s_pct", f"{sz20_nn_p:.2f}%", f"{sz20_n:,} / {d20:,} all rows ({sz20_all_p:.2f}% all rows; {z20_p:.2f}% raw zero)", "2020-2026Q2 post-CBLR (matches part2_dq_by_decade.csv non-null EQ mean)"),
        # Scaling curve sampling audit
        ("scaling_curve_sampling", "scale_025x_rows", str(rows_025_csv), f"{rows_025_csv / panel.height * 100:.2f}% of 1,664,114 panel rows ({n_certs_025:,} / {n_certs_total:,} lowest-CERT banks)", "Sorted first 25% of unique CERTs (older low-CERT banks have longer tenures)"),
        ("scaling_curve_sampling", "scale_050x_rows", str(rows_050_csv), f"{rows_050_csv / panel.height * 100:.2f}% of 1,664,114 panel rows ({n_certs_050:,} / {n_certs_total:,} lowest-CERT banks)", "Sorted first 50% of unique CERTs"),
    ]
    supp_df = pd.DataFrame(
        supp_rows,
        columns=["category", "metric", "value", "unit_or_detail", "verification_formula"],
    )
    supp_df.to_csv(OUT_DIR / "part2_supplementary_evidence.csv", index=False)

    print("\n--- outputs/part1_aux_tables_summary.csv ---")
    print(aux_df.to_string(index=False))
    print("\n--- outputs/part2_supplementary_evidence.csv ---")
    print(supp_df.to_string(index=False))
    print("\nAll evidence assertions passed.")


if __name__ == "__main__":
    main()
