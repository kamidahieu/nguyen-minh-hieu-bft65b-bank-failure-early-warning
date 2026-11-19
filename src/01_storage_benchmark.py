"""
Part 1: Acquisition Verification, Storage Layout & Benchmarking
================================================================
1. Verifies completeness of the FDIC Call Report panel (manifest_financials.json),
   institutions.parquet, and failures.parquet.
2. Documents schema, data types, partitioning (REPDTE=YYYYMMDD), and Snappy compression.
3. Benchmarks disk footprint (MB), full-scan read latency, and filtered query latency
   (6 columns, REPDTE >= '20080331') comparing partitioned Parquet (Polars, PyArrow)
   against a temporary raw CSV export (deleted immediately after benchmarking).
4. Saves results to outputs/part1_storage_benchmark.csv, outputs/part1_schema.csv,
   and outputs/part1_storage_benchmark.png.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

SEED = 42
RAW_DIR = ROOT / "data_raw" / "fdic"
FIN_DIR = RAW_DIR / "financials"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

KEY_STR_COLS = ("REPDTE", "STNAME")
KEY_INT_COLS = ("CERT",)
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

ARROW_SCHEMA = pa.schema(
    [
        pa.field("CERT", pa.int64()),
        pa.field("REPDTE", pa.large_string()),
        pa.field("STNAME", pa.large_string()),
    ]
    + [pa.field(c, pa.float64()) for c in NUMERIC_COLS]
)


def verify_manifest_and_tables() -> dict:
    """Verify manifest completeness, duplicate keys, and auxiliary table counts."""
    manifest_path = RAW_DIR / "manifest_financials.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    quarters = sorted(manifest.keys())
    incomplete = [q for q, m in manifest.items() if not m.get("complete", False)]
    mismatch = [q for q, m in manifest.items() if m.get("rows") != m.get("api_total")]
    manifest_rows = sum(m["rows"] for m in manifest.values())

    # Verify actual parquet files on disk
    parquet_files = sorted(FIN_DIR.glob("REPDTE=*.parquet"))
    keys_df = pl.scan_parquet(str(FIN_DIR / "*.parquet")).select(["CERT", "REPDTE"]).collect()
    n_rows = keys_df.height
    n_banks = keys_df["CERT"].n_unique()
    n_dup = n_rows - keys_df.unique(subset=["CERT", "REPDTE"]).height

    inst_df = pl.read_parquet(RAW_DIR / "institutions.parquet")
    fail_df = pl.read_parquet(RAW_DIR / "failures.parquet")

    # Inspect compression and row-group metadata on a sample partition
    sample_pf = pq.ParquetFile(parquet_files[-1])
    compression = sample_pf.metadata.row_group(0).column(0).compression
    schema_arrow = sample_pf.schema_arrow

    schema_rows = [
        {"column": field.name, "arrow_type": str(field.type), "nullable": field.nullable}
        for field in schema_arrow
    ]
    pd.DataFrame(schema_rows).to_csv(OUT_DIR / "part1_schema.csv", index=False)

    summary = {
        "quarters_in_manifest": len(quarters),
        "parquet_files_on_disk": len(parquet_files),
        "first_quarter": quarters[0],
        "last_quarter": quarters[-1],
        "incomplete_quarters": len(incomplete),
        "api_total_mismatches": len(mismatch),
        "manifest_total_rows": manifest_rows,
        "parquet_total_rows": n_rows,
        "distinct_banks_cert": n_banks,
        "duplicate_cert_repdte": n_dup,
        "institutions_rows": inst_df.height,
        "failures_rows": fail_df.height,
        "num_columns": len(schema_arrow),
        "compression": compression,
    }
    return summary


def dir_size_mb(path: Path) -> float:
    total_bytes = sum(p.stat().st_size for p in path.rglob("*.parquet"))
    return total_bytes / (1024 * 1024)


def bench(fn, repeats: int = 3) -> tuple[float, object]:
    """Run fn `repeats` times, return (median_seconds, last_result)."""
    times = []
    res = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        res = fn()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2], res


def run_storage_benchmark() -> pd.DataFrame:
    """Compare partitioned Parquet (Polars & PyArrow) vs temporary raw CSV."""
    parquet_glob = str(FIN_DIR / "*.parquet")
    parquet_mb = dir_size_mb(FIN_DIR)

    query_cols = ["CERT", "REPDTE", "ASSET", "DEP", "ROA", "NCLNLS"]
    cutoff_repdte = "20080331"

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "financials_all.csv"
        # Materialize raw CSV from the exact same dataset for an apples-to-apples comparison
        t_write0 = time.perf_counter()
        full_pl = pl.scan_parquet(
            parquet_glob, schema=POLARS_SCHEMA, cast_options=SCAN_CAST_OPTS
        ).collect()
        full_pl.write_csv(csv_path)
        csv_write_s = time.perf_counter() - t_write0
        csv_mb = csv_path.stat().st_size / (1024 * 1024)
        del full_pl

        # 1. Full-scan read latency
        t_pq_polars_full, df_a = bench(
            lambda: pl.scan_parquet(
                parquet_glob, schema=POLARS_SCHEMA, cast_options=SCAN_CAST_OPTS
            ).collect()
        )
        t_pq_arrow_full, df_b = bench(
            lambda: ds.dataset(str(FIN_DIR), format="parquet", schema=ARROW_SCHEMA).to_table()
        )
        t_csv_polars_full, df_c = bench(lambda: pl.read_csv(csv_path, infer_schema_length=10000))
        t_csv_pandas_full, df_d = bench(
            lambda: pd.read_csv(csv_path, dtype={"REPDTE": str, "STNAME": str}, low_memory=False)
        )

        # 2. Filtered + projected query latency (6 cols, REPDTE >= 20080331)
        t_pq_polars_filt, res_pq_pl = bench(
            lambda: (
                pl.scan_parquet(parquet_glob, schema=POLARS_SCHEMA, cast_options=SCAN_CAST_OPTS)
                .filter(pl.col("REPDTE") >= cutoff_repdte)
                .select(query_cols)
                .collect()
            )
        )
        t_pq_arrow_filt, res_pq_pa = bench(
            lambda: (
                ds.dataset(str(FIN_DIR), format="parquet", schema=ARROW_SCHEMA)
                .to_table(columns=query_cols, filter=ds.field("REPDTE") >= cutoff_repdte)
            )
        )
        t_csv_polars_filt, res_csv_pl = bench(
            lambda: (
                pl.scan_csv(csv_path, schema_overrides={"REPDTE": pl.Utf8})
                .filter(pl.col("REPDTE") >= cutoff_repdte)
                .select(query_cols)
                .collect()
            )
        )
        t_csv_pandas_filt, res_csv_pd = bench(
            lambda: (
                lambda d: d[d["REPDTE"] >= cutoff_repdte]
            )(pd.read_csv(csv_path, usecols=query_cols, dtype={"REPDTE": str}))
        )

        # Sanity check row counts match across engines
        assert res_pq_pl.height == res_pq_pa.num_rows == res_csv_pl.height == len(res_csv_pd), (
            f"Row count mismatch: {res_pq_pl.height}, {res_pq_pa.num_rows}, "
            f"{res_csv_pl.height}, {len(res_csv_pd)}"
        )
        filtered_rows = res_pq_pl.height
        total_rows = df_a.height
        # Temp CSV is automatically deleted on context exit

    rows = [
        {
            "format": "Partitioned Parquet (Snappy)",
            "engine": "Polars",
            "disk_mb": round(parquet_mb, 2),
            "compression_ratio_vs_csv": round(csv_mb / parquet_mb, 2),
            "full_scan_s": round(t_pq_polars_full, 4),
            "filtered_query_s": round(t_pq_polars_filt, 4),
            "speedup_filtered_vs_pandas_csv": round(t_csv_pandas_filt / t_pq_polars_filt, 2),
            "total_rows": total_rows,
            "filtered_rows": filtered_rows,
        },
        {
            "format": "Partitioned Parquet (Snappy)",
            "engine": "PyArrow Dataset",
            "disk_mb": round(parquet_mb, 2),
            "compression_ratio_vs_csv": round(csv_mb / parquet_mb, 2),
            "full_scan_s": round(t_pq_arrow_full, 4),
            "filtered_query_s": round(t_pq_arrow_filt, 4),
            "speedup_filtered_vs_pandas_csv": round(t_csv_pandas_filt / t_pq_arrow_filt, 2),
            "total_rows": total_rows,
            "filtered_rows": filtered_rows,
        },
        {
            "format": "Raw CSV (uncompressed)",
            "engine": "Polars",
            "disk_mb": round(csv_mb, 2),
            "compression_ratio_vs_csv": 1.0,
            "full_scan_s": round(t_csv_polars_full, 4),
            "filtered_query_s": round(t_csv_polars_filt, 4),
            "speedup_filtered_vs_pandas_csv": round(t_csv_pandas_filt / t_csv_polars_filt, 2),
            "total_rows": total_rows,
            "filtered_rows": filtered_rows,
        },
        {
            "format": "Raw CSV (uncompressed)",
            "engine": "pandas",
            "disk_mb": round(csv_mb, 2),
            "compression_ratio_vs_csv": 1.0,
            "full_scan_s": round(t_csv_pandas_full, 4),
            "filtered_query_s": round(t_csv_pandas_filt, 4),
            "speedup_filtered_vs_pandas_csv": 1.0,
            "total_rows": total_rows,
            "filtered_rows": filtered_rows,
        },
    ]
    bench_df = pd.DataFrame(rows)
    bench_df.to_csv(OUT_DIR / "part1_storage_benchmark.csv", index=False)
    print(f"Temp CSV ({csv_mb:.1f} MB, written in {csv_write_s:.2f}s) deleted cleanly.")
    return bench_df


def plot_benchmark(bench_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), dpi=200)

    # 1. Disk footprint
    fmt_df = bench_df.drop_duplicates("format")
    axes[0].bar(fmt_df["format"], fmt_df["disk_mb"], color=["#1f77b4", "#d62728"])
    axes[0].set_title("Disk Footprint (MB)")
    axes[0].set_ylabel("Size (MB)")
    axes[0].tick_params(axis="x", rotation=15)
    for i, v in enumerate(fmt_df["disk_mb"]):
        axes[0].text(i, v + 5, f"{v:.1f} MB", ha="center", fontsize=9)

    # 2. Full-scan read latency
    labels = [f"{r['format'].split()[1]}\n({r['engine']})" for _, r in bench_df.iterrows()]
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
    axes[1].bar(labels, bench_df["full_scan_s"], color=colors)
    axes[1].set_title("Full-Scan Read Latency (1.68M rows)")
    axes[1].set_ylabel("Seconds (median of 3)")
    for i, v in enumerate(bench_df["full_scan_s"]):
        axes[1].text(i, v + 0.05, f"{v:.2f}s", ha="center", fontsize=9)

    # 3. Filtered + projected query latency
    axes[2].bar(labels, bench_df["filtered_query_s"], color=colors)
    axes[2].set_title("Filtered Query (6 cols, >=2008Q1)")
    axes[2].set_ylabel("Seconds (median of 3)")
    for i, v in enumerate(bench_df["filtered_query_s"]):
        axes[2].text(i, v + 0.03, f"{v:.2f}s", ha="center", fontsize=9)

    fig.suptitle("Part 1: Partitioned Parquet (Snappy) vs. Raw CSV Storage Benchmark", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "part1_storage_benchmark.png", dpi=200)
    plt.close(fig)


def main() -> None:
    print("=== Part 1: Manifest & Completeness Verification ===")
    summary = verify_manifest_and_tables()
    for k, v in summary.items():
        print(f"  {k:<26}: {v}")
    pd.DataFrame([summary]).to_csv(OUT_DIR / "part1_manifest_summary.csv", index=False)

    print("\n=== Part 1: Storage & Query Benchmark ===")
    bench_df = run_storage_benchmark()
    print(bench_df.to_string(index=False))

    plot_benchmark(bench_df)
    print(f"\nSaved outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
