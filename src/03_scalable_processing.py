"""
Part 3: Scalable Panel Processing (Polars vs. pandas vs. PySpark)
=================================================================
1. Implements 16 core panel transformations (point-in-time ratios, 1Q/4Q lags &
   growth rates, 4-quarter rolling means/std devs per contiguous bank spell, and
   cross-sectional quarter demeaning) using Polars LazyFrame.
2. Exports Polars unoptimized and optimized query plans (outputs/part3_polars_explain.txt).
3. Benchmarks Polars (in-memory parallel & streaming engines) fairly against an
   identical pandas baseline and PySpark (local[*]) on wall-clock latency and
   peak memory allocation (tracemalloc & psutil RSS delta).
4. Reproduces all 16 transformations in PySpark, exports Catalyst/Tungsten physical
   execution plan (outputs/part3_spark_explain.txt), and verifies strict numerical
   equivalence against Polars (max_abs_diff < 1e-6 across all 16 columns).
5. Runs an empirical scaling experiment (0.25x to 4.0x panel size, up to ~6.66M rows)
   to quantify single-node throughput limits and when a multi-node Spark cluster
   is justified.
"""
from __future__ import annotations

import gc
import os
import sys
import time
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("PYSPARK_PYTHON", "python")
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", "python")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import psutil

SEED = 42
CLEAN_DIR = ROOT / "data_clean"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PANEL_PATH = CLEAN_DIR / "panel_labeled.parquet"

CORE_INPUT_COLS = [
    "CERT", "REPDTE", "q_idx", "spell_id",
    "ASSET", "DEP", "EQ", "CHBAL", "SC",
    "LNLSGR", "LNATRES", "NCLNLS", "ORE",
    "ROA_win", "NIMY_win",
]

ENGINEERED_COLS = [
    "eq_to_asset",
    "npl_to_loans",
    "reserve_to_loans",
    "loan_to_dep",
    "liquid_to_asset",
    "ore_to_asset",
    "asset_growth_1q",
    "asset_growth_4q",
    "eq_to_asset_chg_4q",
    "roa_chg_4q",
    "npl_to_loans_chg_4q",
    "roa_roll4_mean",
    "roa_roll4_std",
    "nim_roll4_mean",
    "nim_roll4_std",
    "roa_rel_quarter",
]


# ----------------------------------------------------------------------------
# 1. Polars LazyFrame Implementation
# ----------------------------------------------------------------------------
def build_polars_lazy_pipeline(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Build lazy query plan for the 16 core panel transformations."""
    grp = ["CERT", "spell_id"]
    return (
        lf.select(CORE_INPUT_COLS)
        .sort(["CERT", "spell_id", "q_idx"])
        # Stage A: Point-in-time balance sheet & asset quality ratios
        .with_columns(
            [
                (pl.col("EQ") / pl.col("ASSET") * 100.0).alias("eq_to_asset"),
                pl.when(pl.col("LNLSGR") > 0)
                .then(pl.col("NCLNLS") / pl.col("LNLSGR") * 100.0)
                .otherwise(None)
                .alias("npl_to_loans"),
                pl.when(pl.col("LNLSGR") > 0)
                .then(pl.col("LNATRES") / pl.col("LNLSGR") * 100.0)
                .otherwise(None)
                .alias("reserve_to_loans"),
                pl.when(pl.col("DEP") > 0)
                .then(pl.col("LNLSGR") / pl.col("DEP") * 100.0)
                .otherwise(None)
                .alias("loan_to_dep"),
                ((pl.col("CHBAL") + pl.col("SC")) / pl.col("ASSET") * 100.0).alias("liquid_to_asset"),
                (pl.col("ORE") / pl.col("ASSET") * 100.0).alias("ore_to_asset"),
            ]
        )
        # Stage B: 1Q & 4Q lags, growth rates, and 4Q rolling statistics within contiguous (CERT, spell_id)
        .with_columns(
            [
                (
                    (pl.col("ASSET") - pl.col("ASSET").shift(1).over(grp))
                    / pl.col("ASSET").shift(1).over(grp)
                    * 100.0
                ).alias("asset_growth_1q"),
                (
                    (pl.col("ASSET") - pl.col("ASSET").shift(4).over(grp))
                    / pl.col("ASSET").shift(4).over(grp)
                    * 100.0
                ).alias("asset_growth_4q"),
                (pl.col("eq_to_asset") - pl.col("eq_to_asset").shift(4).over(grp)).alias("eq_to_asset_chg_4q"),
                (pl.col("ROA_win") - pl.col("ROA_win").shift(4).over(grp)).alias("roa_chg_4q"),
                (pl.col("npl_to_loans") - pl.col("npl_to_loans").shift(4).over(grp)).alias("npl_to_loans_chg_4q"),
                pl.col("ROA_win")
                .rolling_mean(window_size=4, min_samples=4)
                .over(grp)
                .alias("roa_roll4_mean"),
                pl.col("ROA_win")
                .rolling_std(window_size=4, min_samples=4, ddof=1)
                .over(grp)
                .alias("roa_roll4_std"),
                pl.col("NIMY_win")
                .rolling_mean(window_size=4, min_samples=4)
                .over(grp)
                .alias("nim_roll4_mean"),
                pl.col("NIMY_win")
                .rolling_std(window_size=4, min_samples=4, ddof=1)
                .over(grp)
                .alias("nim_roll4_std"),
                # Stage C: Contemporaneous cross-sectional quarter demeaning
                (pl.col("ROA_win") - pl.col("ROA_win").mean().over("REPDTE")).alias("roa_rel_quarter"),
            ]
        )
    )


# ----------------------------------------------------------------------------
# 2. pandas Baseline Implementation
# ----------------------------------------------------------------------------
def run_pandas_pipeline(pdf: pd.DataFrame) -> pd.DataFrame:
    """Identical panel transformations using idiomatic vectorized pandas."""
    df = pdf[CORE_INPUT_COLS].sort_values(["CERT", "spell_id", "q_idx"]).reset_index(drop=True)

    # Stage A: Point-in-time financial ratios
    df["eq_to_asset"] = (df["EQ"] / df["ASSET"]) * 100.0
    df["npl_to_loans"] = np.where(df["LNLSGR"] > 0, (df["NCLNLS"] / df["LNLSGR"]) * 100.0, np.nan)
    df["reserve_to_loans"] = np.where(df["LNLSGR"] > 0, (df["LNATRES"] / df["LNLSGR"]) * 100.0, np.nan)
    df["loan_to_dep"] = np.where(df["DEP"] > 0, (df["LNLSGR"] / df["DEP"]) * 100.0, np.nan)
    df["liquid_to_asset"] = ((df["CHBAL"] + df["SC"]) / df["ASSET"]) * 100.0
    df["ore_to_asset"] = (df["ORE"] / df["ASSET"]) * 100.0

    # Stage B: Groupby lags and 4-quarter rolling statistics
    g = df.groupby(["CERT", "spell_id"], sort=False)
    asset_lag1 = g["ASSET"].shift(1)
    asset_lag4 = g["ASSET"].shift(4)
    df["asset_growth_1q"] = ((df["ASSET"] - asset_lag1) / asset_lag1) * 100.0
    df["asset_growth_4q"] = ((df["ASSET"] - asset_lag4) / asset_lag4) * 100.0
    df["eq_to_asset_chg_4q"] = df["eq_to_asset"] - g["eq_to_asset"].shift(4)
    df["roa_chg_4q"] = df["ROA_win"] - g["ROA_win"].shift(4)
    df["npl_to_loans_chg_4q"] = df["npl_to_loans"] - g["npl_to_loans"].shift(4)

    df["roa_roll4_mean"] = g["ROA_win"].rolling(4, min_periods=4).mean().reset_index(level=[0, 1], drop=True)
    df["roa_roll4_std"] = g["ROA_win"].rolling(4, min_periods=4).std(ddof=1).reset_index(level=[0, 1], drop=True)
    df["nim_roll4_mean"] = g["NIMY_win"].rolling(4, min_periods=4).mean().reset_index(level=[0, 1], drop=True)
    df["nim_roll4_std"] = g["NIMY_win"].rolling(4, min_periods=4).std(ddof=1).reset_index(level=[0, 1], drop=True)

    # Stage C: Contemporaneous cross-sectional quarter demeaning
    df["roa_rel_quarter"] = df["ROA_win"] - df.groupby("REPDTE", sort=False)["ROA_win"].transform("mean")
    return df


# ----------------------------------------------------------------------------
# 3. PySpark Implementation
# ----------------------------------------------------------------------------
def build_spark_pipeline(sdf):
    """Identical panel transformations using PySpark DataFrame & Window API."""
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    w_order = Window.partitionBy("CERT", "spell_id").orderBy("q_idx")
    w_roll4 = w_order.rowsBetween(-3, 0)
    w_quarter = Window.partitionBy("REPDTE")

    sdf = (
        sdf.select(*CORE_INPUT_COLS)
        .withColumn("eq_to_asset", (F.col("EQ") / F.col("ASSET")) * F.lit(100.0))
        .withColumn(
            "npl_to_loans",
            F.when(F.col("LNLSGR") > 0, (F.col("NCLNLS") / F.col("LNLSGR")) * F.lit(100.0)).otherwise(F.lit(None)),
        )
        .withColumn(
            "reserve_to_loans",
            F.when(F.col("LNLSGR") > 0, (F.col("LNATRES") / F.col("LNLSGR")) * F.lit(100.0)).otherwise(F.lit(None)),
        )
        .withColumn(
            "loan_to_dep",
            F.when(F.col("DEP") > 0, (F.col("LNLSGR") / F.col("DEP")) * F.lit(100.0)).otherwise(F.lit(None)),
        )
        .withColumn("liquid_to_asset", ((F.col("CHBAL") + F.col("SC")) / F.col("ASSET")) * F.lit(100.0))
        .withColumn("ore_to_asset", (F.col("ORE") / F.col("ASSET")) * F.lit(100.0))
    )

    asset_lag1 = F.lag("ASSET", 1).over(w_order)
    asset_lag4 = F.lag("ASSET", 4).over(w_order)

    sdf = (
        sdf.withColumn("asset_growth_1q", ((F.col("ASSET") - asset_lag1) / asset_lag1) * F.lit(100.0))
        .withColumn("asset_growth_4q", ((F.col("ASSET") - asset_lag4) / asset_lag4) * F.lit(100.0))
        .withColumn("eq_to_asset_chg_4q", F.col("eq_to_asset") - F.lag("eq_to_asset", 4).over(w_order))
        .withColumn("roa_chg_4q", F.col("ROA_win") - F.lag("ROA_win", 4).over(w_order))
        .withColumn("npl_to_loans_chg_4q", F.col("npl_to_loans") - F.lag("npl_to_loans", 4).over(w_order))
        .withColumn(
            "roa_roll4_mean",
            F.when(F.count("ROA_win").over(w_roll4) == 4, F.avg("ROA_win").over(w_roll4)).otherwise(F.lit(None)),
        )
        .withColumn(
            "roa_roll4_std",
            F.when(F.count("ROA_win").over(w_roll4) == 4, F.stddev_samp("ROA_win").over(w_roll4)).otherwise(F.lit(None)),
        )
        .withColumn(
            "nim_roll4_mean",
            F.when(F.count("NIMY_win").over(w_roll4) == 4, F.avg("NIMY_win").over(w_roll4)).otherwise(F.lit(None)),
        )
        .withColumn(
            "nim_roll4_std",
            F.when(F.count("NIMY_win").over(w_roll4) == 4, F.stddev_samp("NIMY_win").over(w_roll4)).otherwise(F.lit(None)),
        )
        .withColumn("roa_rel_quarter", F.col("ROA_win") - F.avg("ROA_win").over(w_quarter))
    )
    return sdf


# ----------------------------------------------------------------------------
# Measurement & Verification Helpers
# ----------------------------------------------------------------------------
def total_process_tree_rss_mb() -> float:
    """Return RSS memory (MB) of current Python process plus any child processes (e.g., Spark JVM)."""
    proc = psutil.Process(os.getpid())
    total = proc.memory_info().rss
    for child in proc.children(recursive=True):
        try:
            total += child.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total / (1024 * 1024)


def measure_execution(fn) -> tuple[float, float, float, object]:
    """Run fn(), returning (wall_seconds, tracemalloc_peak_mb, rss_delta_mb, result)."""
    gc.collect()
    rss_before = total_process_tree_rss_mb()
    tracemalloc.start()
    t0 = time.perf_counter()
    res = fn()
    elapsed = time.perf_counter() - t0
    _, trace_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = total_process_tree_rss_mb()
    return elapsed, trace_peak / (1024 * 1024), max(0.0, rss_after - rss_before), res


def verify_numerical_equivalence(
    pl_df: pl.DataFrame, pd_df: pd.DataFrame, spark_pd: pd.DataFrame
) -> pd.DataFrame:
    """Compare Polars against pandas and PySpark across all 16 engineered columns."""
    pl_sorted = pl_df.sort(["CERT", "REPDTE"])
    pd_sorted = pd_df.sort_values(["CERT", "REPDTE"]).reset_index(drop=True)
    sp_sorted = spark_pd.sort_values(["CERT", "REPDTE"]).reset_index(drop=True)

    rows = []
    for col in ENGINEERED_COLS:
        a = pl_sorted[col].to_numpy()
        b_pd = pd_sorted[col].to_numpy(dtype=np.float64)
        b_sp = sp_sorted[col].to_numpy(dtype=np.float64)

        valid_a = ~np.isnan(a)
        valid_pd = ~np.isnan(b_pd)
        valid_sp = ~np.isnan(b_sp)

        null_match_pd = bool(np.array_equal(valid_a, valid_pd))
        null_match_sp = bool(np.array_equal(valid_a, valid_sp))

        diff_pd = np.abs(a[valid_a & valid_pd] - b_pd[valid_a & valid_pd])
        diff_sp = np.abs(a[valid_a & valid_sp] - b_sp[valid_a & valid_sp])

        max_diff_pd = float(diff_pd.max()) if len(diff_pd) > 0 else 0.0
        max_diff_sp = float(diff_sp.max()) if len(diff_sp) > 0 else 0.0
        mean_diff_sp = float(diff_sp.mean()) if len(diff_sp) > 0 else 0.0

        assert null_match_pd and max_diff_pd < 1e-6, f"pandas mismatch on {col}: max_diff={max_diff_pd}"
        assert null_match_sp and max_diff_sp < 1e-6, f"PySpark mismatch on {col}: max_diff={max_diff_sp}"

        rows.append(
            {
                "column": col,
                "non_null_count": int(valid_a.sum()),
                "null_count": int((~valid_a).sum()),
                "polars_vs_pandas_max_abs_diff": max_diff_pd,
                "polars_vs_spark_max_abs_diff": max_diff_sp,
                "polars_vs_spark_mean_abs_diff": mean_diff_sp,
                "equivalence_passed_1e6": bool(max_diff_sp < 1e-6 and null_match_sp),
            }
        )
    eq_df = pd.DataFrame(rows)
    eq_df.to_csv(OUT_DIR / "part3_equivalence_check.csv", index=False)
    return eq_df


def make_scaled_polars_df(base_df: pl.DataFrame, factor: float) -> pl.DataFrame:
    """Create a scaled panel of size `factor * len(base_df)` preserving bank panel structure."""
    if factor < 1.0:
        certs = base_df["CERT"].unique().sort()
        keep_n = int(len(certs) * factor)
        keep_certs = certs.head(keep_n)
        return base_df.filter(pl.col("CERT").is_in(keep_certs.implode()))
    elif factor == 1.0:
        return base_df
    else:
        n_copies = int(round(factor))
        chunks = [base_df]
        for k in range(1, n_copies):
            chunks.append(base_df.with_columns((pl.col("CERT") + k * 1_000_000).alias("CERT")))
        return pl.concat(chunks, how="vertical")


def plot_part3_benchmarks(bench_df: pd.DataFrame, scale_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.3), dpi=200)

    # Panel A: 1.0x Wall-Clock Latency
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
    axes[0].bar(bench_df["engine"], bench_df["wall_time_s"], color=colors[: len(bench_df)])
    axes[0].set_title("A. Execution Time (1.66M Rows, 16 Features)")
    axes[0].set_ylabel("Wall-Clock Time (seconds)")
    axes[0].tick_params(axis="x", rotation=15)
    for i, v in enumerate(bench_df["wall_time_s"]):
        axes[0].text(i, v + 0.1, f"{v:.2f}s", ha="center", fontsize=8.5)

    # Panel B: Peak Memory Overhead
    axes[1].bar(bench_df["engine"], bench_df["peak_memory_mb"], color=colors[: len(bench_df)])
    axes[1].set_title("B. Peak Memory Footprint Delta (MB)")
    axes[1].set_ylabel("Memory Delta (MB)")
    axes[1].tick_params(axis="x", rotation=15)
    for i, v in enumerate(bench_df["peak_memory_mb"]):
        axes[1].text(i, v + 10, f"{v:.0f} MB", ha="center", fontsize=8.5)

    # Panel C: Empirical Scaling Curve (0.25x to 4.0x)
    for eng, col in [("Polars Lazy (Parallel)", "#1f77b4"), ("pandas Baseline", "#ff7f0e"), ("PySpark (local[*])", "#d62728")]:
        sub = scale_df[scale_df["engine"] == eng]
        if not sub.empty:
            axes[2].plot(
                sub["rows"] / 1e6,
                sub["wall_time_s"],
                marker="o",
                linewidth=1.8,
                color=col,
                label=eng,
            )
    axes[2].set_title("C. Scaling Curve (0.42M to 6.66M Rows)")
    axes[2].set_xlabel("Panel Rows (Millions)")
    axes[2].set_ylabel("Wall-Clock Time (seconds)")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Part 3: Scalable Panel Processing Benchmark (Polars vs. pandas vs. PySpark)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "part3_benchmarks.png", dpi=200)
    plt.close(fig)


def main() -> None:
    print("=== Part 3: Polars LazyFrame Query Plan & Execution ===")
    lf_raw = pl.scan_parquet(str(PANEL_PATH))
    lf_plan = build_polars_lazy_pipeline(lf_raw)

    unopt_plan = lf_plan.explain(optimized=False)
    opt_plan = lf_plan.explain(optimized=True)
    plan_text = (
        "=== POLARS UNOPTIMIZED LOGICAL PLAN ===\n"
        + unopt_plan
        + "\n\n=== POLARS OPTIMIZED PHYSICAL PLAN ===\n"
        + opt_plan
        + "\n"
    )
    (OUT_DIR / "part3_polars_explain.txt").write_text(plan_text, encoding="utf-8")
    print("Saved Polars logical & optimized query plans to outputs/part3_polars_explain.txt")
    print(opt_plan)

    # Pre-load base input columns into memory so transformation benchmarks compare compute fairly,
    # and also measure end-to-end from disk scan for Polars Lazy
    base_pl = pl.read_parquet(PANEL_PATH, columns=CORE_INPUT_COLS).with_columns(
        pl.col("spell_id").cast(pl.Int64)
    )
    base_pd = base_pl.to_pandas()
    n_rows = base_pl.height

    # 1. Benchmark Polars Lazy (in-memory parallel)
    t_pl, tm_pl, rss_pl, res_pl = measure_execution(
        lambda: build_polars_lazy_pipeline(base_pl.lazy()).collect()
    )

    # 2. Benchmark Polars Lazy (streaming engine from disk scan)
    t_pl_str, tm_pl_str, rss_pl_str, res_pl_str = measure_execution(
        lambda: build_polars_lazy_pipeline(pl.scan_parquet(str(PANEL_PATH))).collect(engine="streaming")
    )

    # 3. Benchmark pandas Baseline
    t_pd, tm_pd, rss_pd, res_pd = measure_execution(lambda: run_pandas_pipeline(base_pd))

    # 4. Benchmark PySpark (local[*])
    print("\n=== Part 3: PySpark Execution & Physical Plan ===")
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.master("local[*]")
        .appName("BFT65B-Part3-ScalableProcessing")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.shuffle.partitions", "16")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    # Load data into Spark DataFrame and cache input so compute transformation latency is measured fairly
    try:
        sdf_in = spark.read.parquet(str(PANEL_PATH)).select(*CORE_INPUT_COLS)
        sdf_in.cache()
        sdf_in.count()
    except Exception:
        sdf_in = spark.createDataFrame(base_pd)
        sdf_in.cache()
        sdf_in.count()

    sdf_out = build_spark_pipeline(sdf_in)
    spark_explain = sdf_out._jdf.queryExecution().toString()
    (OUT_DIR / "part3_spark_explain.txt").write_text(spark_explain, encoding="utf-8")
    print("Saved Spark Catalyst/Tungsten execution plan to outputs/part3_spark_explain.txt")

    def run_spark_action():
        df_s = build_spark_pipeline(sdf_in)
        df_s.write.format("noop").mode("overwrite").save()
        return df_s

    t_sp, tm_sp, rss_sp, _ = measure_execution(run_spark_action)
    # Capture total JVM child RSS for PySpark
    jvm_rss_mb = sum(
        c.memory_info().rss for c in psutil.Process(os.getpid()).children(recursive=True)
    ) / (1024 * 1024)
    res_spark_pd = sdf_out.toPandas()

    print("\n=== Part 3: Numerical Equivalence Verification (Polars vs. pandas vs. PySpark) ===")
    eq_df = verify_numerical_equivalence(res_pl, res_pd, res_spark_pd)
    print(eq_df.to_string(index=False))

    # Estimate Polars output Arrow buffer size in MB when rss_delta is small due to allocator pooling
    pl_out_mb = res_pl.estimated_size("mb")

    bench_rows = [
        {
            "engine": "Polars Lazy (Parallel)",
            "rows": n_rows,
            "wall_time_s": round(t_pl, 4),
            "speedup_vs_pandas": round(t_pd / t_pl, 2),
            "peak_memory_mb": round(max(tm_pl, rss_pl, pl_out_mb), 1),
        },
        {
            "engine": "Polars Lazy (Streaming)",
            "rows": n_rows,
            "wall_time_s": round(t_pl_str, 4),
            "speedup_vs_pandas": round(t_pd / t_pl_str, 2),
            "peak_memory_mb": round(max(tm_pl_str, rss_pl_str, pl_out_mb), 1),
        },
        {
            "engine": "pandas Baseline",
            "rows": n_rows,
            "wall_time_s": round(t_pd, 4),
            "speedup_vs_pandas": 1.0,
            "peak_memory_mb": round(max(tm_pd, rss_pd), 1),
        },
        {
            "engine": "PySpark (local[*])",
            "rows": n_rows,
            "wall_time_s": round(t_sp, 4),
            "speedup_vs_pandas": round(t_pd / t_sp, 2),
            "peak_memory_mb": round(max(tm_sp, rss_sp, jvm_rss_mb), 1),
        },
    ]
    bench_df = pd.DataFrame(bench_rows)
    bench_df.to_csv(OUT_DIR / "part3_benchmarks.csv", index=False)
    print("\n--- Engine Benchmark Summary (1.0x Panel = 1,664,114 Rows) ---")
    print(bench_df.to_string(index=False))

    # 5. Empirical Scaling Curve (0.25x, 0.5x, 1.0x, 2.0x, 4.0x)
    print("\n=== Part 3: Empirical Scaling Experiment (0.25x to 4.0x Panel Size) ===")
    from pyspark.sql import functions as F

    scale_rows = []
    for factor in (0.25, 0.5, 1.0, 2.0, 4.0):
        scaled_pl = make_scaled_polars_df(base_pl, factor)
        s_rows = scaled_pl.height
        t_s_pl, _, rss_s_pl, _ = measure_execution(
            lambda: build_polars_lazy_pipeline(scaled_pl.lazy()).collect()
        )
        scale_rows.append(
            {"scale_factor": factor, "rows": s_rows, "engine": "Polars Lazy (Parallel)", "wall_time_s": round(t_s_pl, 4)}
        )

        scaled_pd = scaled_pl.to_pandas()
        t_s_pd, tm_s_pd, _, _ = measure_execution(lambda: run_pandas_pipeline(scaled_pd))
        scale_rows.append(
            {"scale_factor": factor, "rows": s_rows, "engine": "pandas Baseline", "wall_time_s": round(t_s_pd, 4)}
        )

        # Build scaled Spark DataFrame natively on the JVM to measure pure engine scaling
        if factor < 1.0:
            max_cert = int(scaled_pl["CERT"].max())
            sdf_scaled = sdf_in.filter(F.col("CERT") <= F.lit(max_cert))
        elif factor == 1.0:
            sdf_scaled = sdf_in
        else:
            n_copies = int(round(factor))
            sdf_scaled = sdf_in
            for k in range(1, n_copies):
                sdf_scaled = sdf_scaled.unionByName(
                    sdf_in.withColumn("CERT", F.col("CERT") + F.lit(k * 1_000_000))
                )
        sdf_scaled.cache()
        sdf_scaled.count()
        t0_sp = time.perf_counter()
        build_spark_pipeline(sdf_scaled).write.format("noop").mode("overwrite").save()
        t_s_sp = time.perf_counter() - t0_sp
        if factor != 1.0:
            sdf_scaled.unpersist()
        scale_rows.append(
            {"scale_factor": factor, "rows": s_rows, "engine": "PySpark (local[*])", "wall_time_s": round(t_s_sp, 4)}
        )
        del scaled_pl, scaled_pd
        gc.collect()

    spark.stop()

    scale_df = pd.DataFrame(scale_rows)
    scale_df.to_csv(OUT_DIR / "part3_scaling_curve.csv", index=False)
    print(scale_df.to_string(index=False))

    plot_part3_benchmarks(bench_df, scale_df)
    print(f"\nSaved all Part 3 outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
