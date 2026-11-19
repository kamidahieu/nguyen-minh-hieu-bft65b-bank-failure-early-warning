# BFT65B Final Individual Report — A Big-Data Pipeline for Bank Failure Early Warning

End-to-end, reproducible data-engineering and early-warning modeling pipeline built on 42 years of US FDIC Call Report, Institution, and Failure data (`1984Q1–2026Q2`, `1,682,738` bank-quarters, `24,064` distinct banks).

---

## 1. Key Scope & Summary Metrics (First-Page Report Reference)

| Attribute | Value |
|---|---|
| **Data Source & Acquisition Method** | Public FDIC BankFind Suite API (`https://api.fdic.gov/banks`) via resumable quarter-partitioned downloader [`fdic_download_starter.py`](fdic_download_starter.py) |
| **Raw Financial Panel Scope** | `1984Q1–2026Q2` (`170` quarters, `1,682,738` rows, `24,064` distinct `CERT` banks, `3` keys + `30` Call Report fields, `0` duplicates) |
| **Auxiliary Tables** | `institutions.parquet` (`27,835` rows, `9` fields) and `failures.parquet` (`4,117` rows, `9` fields; `3,275` events $\ge$ 1984 across `3,240` unique `CERT`s) |
| **Post-Failure Conservatorship Cleaning** | `1,095` Call Reports filed *on or after* a bank's post-1984 outright failure date (`rep_dt >= fail_dt`, 1985–1992 S&L conservatorships) excluded |
| **Clean Labeled Modeling Panel ($\le$ 2025Q2)** | `1,664,114` bank-quarters (`166` quarters, `1984Q1–2025Q2`); `2025Q3–2026Q2` (`17,529` rows) excluded due to right-censored 4Q labels |
| **Development Period (`split == 'dev'`)** | **`1984Q1–2006Q4`** (`92` quarters, `1,193,914` rows, `23,624` distinct banks, **`10,251` positive labels $y_{i,q}=1$ [0.859%]**, **`2,658` distinct failing banks**) |
| **Embargo / Purge Gap (`split == 'embargo'`)** | **`2007Q1–2007Q4`** (`4` quarters, `34,754` rows, `53` positive labels, `32` failing banks) — prevents 4-quarter forward label leakage into Test |
| **Held-Out Test Period (`split == 'test'`)** | **`2008Q1–2025Q2`** (`70` quarters, `435,446` rows, `8,798` distinct banks, **`2,181` positive labels $y_{i,q}=1$ [0.501%]**, **`558` distinct failing banks**) |
| **Primary Model Held-Out Test Performance** | `HistGradientBoosting (GBM)`: **PR-AUC = `0.6161`**, **ROC-AUC = `0.9834`**, **Brier = `0.00313`**, **Recall@Top-1% = `77.35%`**, **Recall@Top-5% = `93.76%`**, **Cost Reduction = `74.23%`** (`84.69%` recall at frozen $\tau^*_{\text{dev}} = 0.0180$) |

---

## 2. Repository Structure

```text
.
├── .gitignore                        # Excludes data_raw/, data_clean/, *.parquet while tracking outputs/
├── README.md                         # Setup, end-to-end execution guide, architecture & academic citations
├── requirements.txt                  # Pinned Python dependencies (Python 3.14, Polars 2.1, PySpark 4.3, sklearn 1.9)
├── fdic_download_starter.py          # Resumable FDIC API downloader & manifest completeness verifier
├── src/
│   ├── 01_storage_benchmark.py       # Part 1 [15 pts]: Manifest verification & Partitioned Parquet vs. CSV benchmark
│   ├── 02_data_quality_and_panel.py  # Part 2 [20 pts]: Automated DQ audit, point-in-time cleaning & 4Q failure labels
│   ├── 03_scalable_processing.py     # Part 3 [25 pts]: Polars Lazy vs. pandas vs. PySpark benchmark & scaling curve
│   ├── 04_feature_pipeline.py        # Part 4 [15 pts]: 39 CAMELS features, zero-leakage gate test & sklearn Pipeline
│   └── 05_modeling_and_validation.py # Part 5 [15 pts]: 4-fold Purged CV, empirical cost threshold, MLlib & single Test run
└── outputs/
    ├── part1_manifest_summary.csv    # Completeness verification summary (170 quarters, 1,682,738 rows)
    ├── part1_schema.csv              # Arrow/Parquet column schema & types
    ├── part1_storage_benchmark.csv   # Disk size, full-scan, and filtered query latencies (Parquet vs. CSV)
    ├── part1_storage_benchmark.png   # High-DPI storage & query benchmark figure
    ├── part2_dq_report.csv           # Column-level audit (nulls, zeros, structural zeros, negatives, quantiles)
    ├── part2_dq_by_decade.csv        # Decade-by-decade structural missingness & sentinel zeros (1980s–2020s)
    ├── part2_exit_types.csv          # Bank exit classification (mergers/CHANGEC1 vs. failures vs. active)
    ├── part2_anomaly_checks.csv      # Accounting identity & reporting gap audit
    ├── part2_dev_winsor_bounds.csv   # Point-in-time 0.1%/99.9% ratio bounds fitted strictly on Development (<=2006Q4)
    ├── part2_split_summary.csv       # Row, bank, and failure counts across Dev, Embargo, Test, and Right-Censored splits
    ├── part2_dq_overview.png         # High-DPI structural missingness & 42-year failure rate timeline
    ├── part3_polars_explain.txt      # Polars unoptimized logical plan & optimized physical plan (CSE, pushdown)
    ├── part3_spark_explain.txt       # PySpark Catalyst & Tungsten physical plan (Exchange hashpartitioning, Window)
    ├── part3_equivalence_check.csv   # Numerical equivalence check across all 16 columns (max_abs_diff <= 7.11e-15)
    ├── part3_benchmarks.csv          # Wall-clock latency & memory comparison (Polars Lazy/Streaming vs. pandas vs. Spark)
    ├── part3_scaling_curve.csv       # Empirical scaling measurements from 0.25x (0.52M rows) to 4.0x (6.66M rows)
    ├── part3_benchmarks.png          # High-DPI execution speed, memory, and scaling curve charts
    ├── part4_feature_dictionary.csv  # Full 39-feature CAMELS catalog, formulas, Dev imputation medians & univariate AUC
    ├── part4_pipeline_summary.csv    # Transformed Dev (1,193,914 x 71) and Test (435,446 x 71) matrix diagnostics
    ├── part4_camels_features.png     # High-DPI Top-15 CAMELS feature ranking & pillar discrimination chart
    ├── part5_empirical_cost_params.csv # Asymmetric failure vs. false-alarm costs estimated from Dev failures.parquet
    ├── part5_cv_fold_metrics.csv     # Per-fold metrics across 4 Purged Expanding-Window CV folds on Development
    ├── part5_cv_summary.csv          # Pooled Out-of-Fold Purged CV comparison on Development (1988Q1–2006Q4)
    ├── part5_dev_cost_curves.csv     # Expected supervisory + DIF cost across probability thresholds on Dev OOF
    ├── part5_dev_optimal_thresholds.csv # Optimal alarm thresholds tau* selected strictly on Development OOF
    ├── part5_threshold_sensitivity.csv  # Sensitivity analysis across C_FN:C_FP ratios (10:1, 20:1, 40:1, 50:1)
    ├── part5_test_evaluation.csv     # Single held-out Test evaluation (2008Q1–2025Q2) across sklearn & PySpark MLlib
    ├── part5_test_subperiods.csv     # Test sub-period performance (2008–2014 GFC vs. 2015–2022 Calm vs. 2023 Crisis)
    ├── part5_feature_importance.csv  # Permutation feature importance (PR-AUC drop) on Development data
    ├── part5_model_evaluation.png    # High-DPI 4-panel Test PR curve, ROC curve, Dev cost curve, and feature importance
    └── part6_executive_summary.md    # Evidence-based synthesis for Q1–Q3, limitations, and Vietnam (SBV) adaptation
```

---

## 3. Environment Setup & End-to-End Reproducibility

### Prerequisites
- **Python:** 3.11+ (tested on Python 3.14.0)
- **Java Runtime:** OpenJDK 17 (required for local PySpark execution in Steps 3 and 5; verified with Temurin OpenJDK 17.0.21)

### Installation
```bash
python -m pip install -r requirements.txt
```

### Step 0: Acquire Raw Data (Skipped automatically if `data_raw/fdic/` is already populated)
```bash
python fdic_download_starter.py --start 1984-03-31 --end 2026-06-30 --out data_raw/fdic
```
*(Resumable: checks `data_raw/fdic/manifest_financials.json` and skips any quarter already downloaded and verified).*

### Steps 1–5: Execute the Complete Engineering & Modeling Pipeline
Run each modular script sequentially from the repository root (`SEED = 42` is fixed across all scripts):

```bash
# Part 1 [15 pts]: Verify manifest completeness & benchmark Partitioned Parquet vs. raw CSV
python src/01_storage_benchmark.py

# Part 2 [20 pts]: Run automated DQ audit, point-in-time cleaning & build data_clean/panel_labeled.parquet
python src/02_data_quality_and_panel.py

# Part 3 [25 pts]: Run Polars Lazy vs. pandas vs. PySpark transformations, equivalence check & scaling curve
python src/03_scalable_processing.py

# Part 4 [15 pts]: Engineer 39 CAMELS features, run programmatic zero-leakage test & fit sklearn preprocessor
python src/04_feature_pipeline.py

# Part 5 [15 pts]: Run 4-fold Purged Expanding CV, cost-threshold optimization, PySpark MLlib & single Test evaluation
python src/05_modeling_and_validation.py
```

---

## 4. Academic Integrity, References & AI-Tool Attribution

### Data & Domain References
1. **Federal Deposit Insurance Corporation (FDIC):** *BankFind Suite Public API & RISView Field Dictionary* (`https://api.fdic.gov/banks/docs/risview_properties.yaml`).
2. **Cole, R. A., & White, L. J. (2012):** "Déjà Vu All Over Again: The Causes of U.S. Commercial Bank Failures This Time Around," *Journal of Financial Services Research*, 42(1), 5–29. (Supervisory CAMELS proxies and real-estate concentration predictors).
3. **Wheelock, D. C., & Wilson, P. W. (2000):** "Why Do Banks Disappear? The Determinants of U.S. Bank Failures and Acquisitions," *The Review of Economics and Statistics*, 82(1), 127–138. (Distinguishing outright failures from mergers and non-failure exits).
4. **State Bank of Vietnam (SBV):** Circular 41/2016/TT-NHNN (Basel II Capital Adequacy Ratio regulations) and Law on Credit Institutions 2024 (No. 32/2024/QH15, Early Intervention and Special Control framework).

### Software & AI-Tool Attribution (Section 4.3 Compliance)
- **Starter Code:** Adapted from the course-provided [`fdic_download_starter.py`](fdic_download_starter.py) script (BFT65B – Big Data Analytics in Finance, National Economics University).
- **AI Assistance Disclosure:** AI coding assistant tools (Google Antigravity / Gemini) were used to assist in structuring the modular Python pipeline (`src/01_...` through `src/05_...`), drafting automated data-quality assertions, verifying numerical equivalence between Polars and PySpark execution plans, and formatting summary tables/charts. All pipeline design choices, point-in-time leakage controls, empirical measurements, and economic interpretations were verified against the raw FDIC dataset.
