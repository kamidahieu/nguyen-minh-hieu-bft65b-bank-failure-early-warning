# BFT65B Final Individual Report — A Big-Data Pipeline for Bank Failure Early Warning

End-to-end, reproducible data-engineering and early-warning modeling pipeline built on 42 years of US FDIC Call Report, Institution, and Failure data (`1984Q1–2026Q2`, `1,682,738` bank-quarters, `24,064` distinct banks).

---

## 1. Key Scope & Summary Metrics (First-Page Report Reference)

| Attribute | Value | Source Artifact |
|---|---|---|
| **Data Source & Acquisition Method** | Public FDIC BankFind Suite API (`https://api.fdic.gov/banks`) via resumable quarter-partitioned downloader [`fdic_download_starter.py`](fdic_download_starter.py); downloaded `2026-11-18` (`23:29:02 +07:00` / `16:29:02 UTC` per `manifest_financials.json`) | [`outputs/part1_manifest_summary.csv`](outputs/part1_manifest_summary.csv), [`outputs/part1_aux_tables_summary.csv`](outputs/part1_aux_tables_summary.csv) |
| **Raw Financial Panel Scope** | `1984Q1–2026Q2` (`170` quarters, `1,682,738` rows, `24,064` distinct `CERT` banks, `3` keys + `30` Call Report fields = `33` columns, `0` duplicates; Snappy Parquet `236.76 MB` vs. `422.52 MB` CSV) | [`outputs/part1_manifest_summary.csv`](outputs/part1_manifest_summary.csv), [`outputs/part1_schema.csv`](outputs/part1_schema.csv), [`outputs/part1_storage_benchmark.csv`](outputs/part1_storage_benchmark.csv) |
| **Auxiliary Tables & Failure Reconciliation** | `institutions.parquet` (`27,835` rows, `9` fields) and `failures.parquet` (`4,117` rows, `9` fields: `3,524` `FAILURE` + `593` `ASSISTANCE`; `3,275` events $\ge$ `1984-01-01` across `3,240` unique `CERT`s with `35` banks having `2` events; **`3,218`** appear in Call Reports and **`22`** have no Call Report rows) | [`outputs/part1_aux_tables_summary.csv`](outputs/part1_aux_tables_summary.csv), [`outputs/part2_exit_types.csv`](outputs/part2_exit_types.csv), [`outputs/part2_supplementary_evidence.csv`](outputs/part2_supplementary_evidence.csv) |
| **Post-Failure Conservatorship Cleaning** | `1,095` Call Reports filed *on or after* a bank's post-1984 outright failure date (`rep_dt >= fail_dt`, `RESTYPE == 'FAILURE'`, `1985–1992` across `268` distinct S&L/commercial banks) excluded | [`outputs/part2_supplementary_evidence.csv`](outputs/part2_supplementary_evidence.csv) |
| **Clean Labeled Modeling Panel ($\le$ 2025Q2)** | `1,664,114` bank-quarters (`166` quarters, `1984Q1–2025Q2`); `2025Q3–2026Q2` (`17,529` rows) excluded due to right-censored 4Q labels | [`outputs/part2_split_summary.csv`](outputs/part2_split_summary.csv) |
| **Development Period (`split == 'dev'`)** | **`1984Q1–2006Q4`** (`92` quarters, `1,193,914` rows, `23,624` distinct banks, **`10,251` positive labels $y_{i,q}=1$ [`0.8586%`]**, **`2,658` distinct failing banks**; `EQ < 0` in `41.69%` of failing vs. `0.84%` of non-failing dev rows) | [`outputs/part2_split_summary.csv`](outputs/part2_split_summary.csv), [`outputs/part2_supplementary_evidence.csv`](outputs/part2_supplementary_evidence.csv) |
| **Development Purged OOF CV Window** | **`1988Q1–2006Q4`** (`907,573` pooled out-of-fold validation rows, `6,560` positive labels across 4 expanding-window folds after `1984Q1–1986Q4` warm-up + `1987Q1–1987Q4` 4Q purge) | [`outputs/part5_cv_fold_metrics.csv`](outputs/part5_cv_fold_metrics.csv), [`outputs/part5_cv_summary.csv`](outputs/part5_cv_summary.csv) |
| **Embargo / Purge Gap (`split == 'embargo'`)** | **`2007Q1–2007Q4`** (`4` quarters, `34,754` rows, `8,892` distinct banks, `53` positive labels [`0.1525%`], `32` failing banks) — prevents 4-quarter forward label leakage into Test | [`outputs/part2_split_summary.csv`](outputs/part2_split_summary.csv) |
| **Held-Out Test Period (`split == 'test'`)** | **`2008Q1–2025Q2`** (`70` quarters, `435,446` rows, `8,798` distinct banks, **`2,181` positive labels $y_{i,q}=1$ [`0.5009%`]**, **`558` distinct failing banks**) | [`outputs/part2_split_summary.csv`](outputs/part2_split_summary.csv) |
| **Primary Model Held-Out Test Performance** | `HistGradientBoosting (GBM)`: **PR-AUC = `0.6161`**, **ROC-AUC = `0.9834`**, **Brier = `0.00313`**, **Recall@Top-1% = `77.35%`**, **Recall@Top-5% = `93.76%`**, **Cost Reduction = `74.23%`** (`84.69%` recall at frozen $\tau^*_{\text{dev}} = 0.0180$) | [`outputs/part5_test_evaluation.csv`](outputs/part5_test_evaluation.csv), [`outputs/part5_dev_optimal_thresholds.csv`](outputs/part5_dev_optimal_thresholds.csv), [`outputs/part5_empirical_cost_params.csv`](outputs/part5_empirical_cost_params.csv) |

### Key Reconciliations & Methodological Notes
- **Failure-Bank Reconciliation (`3,240` vs. `3,218`, [`outputs/part2_supplementary_evidence.csv`](outputs/part2_supplementary_evidence.csv)):** `failures.parquet` contains `3,275` post-1984 events (`2,863` `FAILURE` + `412` `ASSISTANCE`) across **`3,240` unique `CERT`s** (with `35` banks experiencing `2` post-1984 events, e.g., open-bank assistance followed by outright failure). Exactly **`3,218`** of those `3,240` `CERT`s appear in the `1984Q1–2026Q2` Call Reports (matching [`outputs/part2_exit_types.csv`](outputs/part2_exit_types.csv)), while **`22` failing `CERT`s have zero Call Report rows** (`16` failed between `1984-01-01` and `1984-03-30` before the first `1984Q1` filing date `1984-03-31`, and `6` had no Call Report filings on or after `1984-03-31`).
- **Development Split (`1984Q1–2006Q4`) vs. Purged CV OOF Window (`1988Q1–2006Q4`, [`outputs/part5_cv_fold_metrics.csv`](outputs/part5_cv_fold_metrics.csv), [`outputs/part5_cv_summary.csv`](outputs/part5_cv_summary.csv)):** While the full Development split spans `1984Q1–2006Q4` (`1,193,914` rows, `10,251` positives), the pooled out-of-fold validation table in `part5_cv_summary.csv` covers **`1988Q1–2006Q4`** (`907,573` rows, `6,560` positives) because expanding-window Fold 1 reserves `1984Q1–1986Q4` as the initial historical training warm-up and `1987Q1–1987Q4` as a strict 4-quarter embargo purge buffer before out-of-fold evaluation starts in `1988Q1`. Final held-out models are then refitted on the entire `1984Q1–2006Q4` Development split prior to the single Test evaluation (`2008Q1–2025Q2`).
- **Scaling Curve Sub-Sampling Note ([`outputs/part3_scaling_curve.csv`](outputs/part3_scaling_curve.csv), [`outputs/part2_supplementary_evidence.csv`](outputs/part2_supplementary_evidence.csv)):** To preserve intact bank time-series spells for lagged/rolling windows, `0.25x` and `0.50x` in `src/03_scalable_processing.py` sample the first `25%` (`6,013`) and `50%` (`12,027`) of sorted unique `CERT`s rather than random rows; because lower FDIC `CERT` numbers belong to older institutions with longer reporting tenures, `0.25x` of `CERT`s yields **`522,516` rows** (`0.52M`, `31.40%` of the `1,664,114`-row panel) and `0.50x` yields **`1,001,948` rows** (`60.21%`), whereas `2.0x` (`3,328,228` rows) and `4.0x` (`6,656,456` rows) replicate full panels with disjoint offset `CERT` IDs.
- **Data Quality Audit Nuance (`RBCRWAJ` Zeroes, [`outputs/part2_supplementary_evidence.csv`](outputs/part2_supplementary_evidence.csv)):** Raw `RBCRWAJ == 0` rates rise from **`0.32%` in 2019** (`68 / 21,366` rows, all with `EQ == null`) to **`41.50%` in 2026** (`3,596 / 8,666` rows) due to the 2020 CBLR exemption; restricting strictly to structural zeroes where `EQ` is non-null and non-zero yields **`0.00%` in 2019** (`0 / 21,366`) vs. **`41.10%` of all 2026 rows** (`3,562 / 8,666`; `41.27%` of non-null `EQ` rows), matching the `100.00%` (1980s) and `36.92%` (2020s) non-null `EQ` rates in [`outputs/part2_dq_by_decade.csv`](outputs/part2_dq_by_decade.csv).

---

## 2. Repository Structure

```text
.
├── .gitignore                            # Excludes data_raw/, data_clean/, *.parquet, .venv/ while tracking outputs/
├── README.md                             # Setup, end-to-end execution guide, architecture & academic citations
├── requirements.txt                      # Pinned Python dependencies (verified 13/13 against outputs/env_versions.txt)
├── fdic_download_starter.py              # Resumable FDIC API downloader & manifest completeness verifier
├── src/
│   ├── 01_storage_benchmark.py           # Part 1 [15 pts]: Manifest verification & Partitioned Parquet vs. CSV benchmark
│   ├── 02_data_quality_and_panel.py      # Part 2 [20 pts]: Automated DQ audit, point-in-time cleaning & 4Q failure labels
│   ├── 03_scalable_processing.py         # Part 3 [25 pts]: Polars Lazy vs. pandas vs. PySpark benchmark & scaling curve
│   ├── 04_feature_pipeline.py            # Part 4 [15 pts]: 39 CAMELS features, zero-leakage gate test & sklearn Pipeline
│   ├── 05_modeling_and_validation.py     # Part 5 [15 pts]: 4-fold Purged CV, empirical cost threshold, MLlib & single Test run
│   └── 06_export_evidence.py             # Audit [Read-Only]: Exports auxiliary table counts, reconciliations & DQ proofs
└── outputs/
    ├── env_versions.txt                  # Exact runtime proof (Python 3.14.10, OpenJDK 17.0.21.1, .venv pip list)
    ├── part1_manifest_summary.csv        # Completeness verification summary (170 quarters, 1,682,738 rows)
    ├── part1_aux_tables_summary.csv      # Row/col counts, RESTYPE split (3,524/593) & manifest download timestamps
    ├── part1_schema.csv                  # Arrow/Parquet column schema & types
    ├── part1_storage_benchmark.csv       # Disk size, full-scan, and filtered query latencies (Parquet vs. CSV)
    ├── part1_storage_benchmark.png       # High-DPI storage & query benchmark figure
    ├── part2_dq_report.csv               # Column-level audit (nulls, zeros, structural zeros, negatives, quantiles)
    ├── part2_dq_by_decade.csv            # Decade-by-decade structural missingness & sentinel zeros (1980s–2020s)
    ├── part2_exit_types.csv              # Bank exit classification (mergers/CHANGEC1 vs. failures vs. active)
    ├── part2_anomaly_checks.csv          # Accounting identity & reporting gap audit
    ├── part2_supplementary_evidence.csv  # Reconciliations (3,240 vs 3,218; 35 multi-event; 1,095 rows; EQ<0; gaps; CBLR)
    ├── part2_dev_winsor_bounds.csv       # Point-in-time 0.1%/99.9% ratio bounds fitted strictly on Development (<=2006Q4)
    ├── part2_split_summary.csv           # Row, bank, and failure counts across Dev, Embargo, Test, and Right-Censored splits
    ├── part2_dq_overview.png             # High-DPI structural missingness & 42-year failure rate timeline
    ├── part3_polars_explain.txt          # Polars unoptimized logical plan & optimized physical plan (CSE, pushdown)
    ├── part3_spark_explain.txt           # PySpark Catalyst & Tungsten physical plan (Exchange hashpartitioning, Window)
    ├── part3_equivalence_check.csv       # Numerical equivalence check across all 16 columns (max_abs_diff <= 7.11e-15)
    ├── part3_benchmarks.csv              # Wall-clock latency & memory comparison (Polars Lazy/Streaming vs. pandas vs. Spark)
    ├── part3_scaling_curve.csv           # Empirical scaling from 0.25x (522,516 rows; lowest 25% CERTs) to 4.0x (6.66M rows)
    ├── part3_benchmarks.png              # High-DPI execution speed, memory, and scaling curve charts
    ├── part4_feature_dictionary.csv      # Full 39-feature CAMELS catalog, formulas, Dev imputation medians & univariate AUC
    ├── part4_pipeline_summary.csv        # Transformed Dev (1,193,914 x 71) and Test (435,446 x 71) matrix diagnostics
    ├── part4_camels_features.png         # High-DPI Top-15 CAMELS feature ranking & pillar discrimination chart
    ├── part5_empirical_cost_params.csv   # Asymmetric failure vs. false-alarm costs estimated from Dev failures.parquet
    ├── part5_cv_fold_metrics.csv         # Per-fold metrics across 4 Purged Expanding-Window CV folds on Development
    ├── part5_cv_summary.csv              # Pooled Out-of-Fold Purged CV comparison on Development (1988Q1–2006Q4)
    ├── part5_dev_cost_curves.csv         # Expected supervisory + DIF cost across probability thresholds on Dev OOF
    ├── part5_dev_optimal_thresholds.csv  # Optimal alarm thresholds tau* selected strictly on Development OOF
    ├── part5_threshold_sensitivity.csv   # Sensitivity analysis across C_FN:C_FP ratios (10:1, 20:1, 40:1, 50:1)
    ├── part5_test_evaluation.csv         # Single held-out Test evaluation (2008Q1–2025Q2) across sklearn & PySpark MLlib
    ├── part5_test_subperiods.csv         # Test sub-period performance (2008–2014 GFC vs. 2015–2022 Calm vs. 2023 Crisis)
    ├── part5_feature_importance.csv      # Permutation feature importance (PR-AUC drop) on Development data
    ├── part5_model_evaluation.png        # High-DPI 4-panel Test PR curve, ROC curve, Dev cost curve, and feature importance
    ├── part6_executive_summary.md        # Evidence-based synthesis for Q1–Q3, limitations, and Vietnam (SBV) adaptation
    └── BFT65B_Final_Report.docx          # Final formatted individual research report submission (DOCX)
```


## 3. Environment Setup & End-to-End Reproducibility

### Prerequisites (Verified in [`outputs/env_versions.txt`](outputs/env_versions.txt))
- **Python:** 3.11+ (verified on **`Python 3.14.10`** in active `.venv`)
- **Java Runtime:** OpenJDK 17 (required for local PySpark execution in Steps 3 and 5; verified with **`Temurin OpenJDK 17.0.21.1+2`**)
- **Pinned Dependencies:** All `13` packages pinned in [`requirements.txt`](requirements.txt) match the active `.venv` `pip list` (`0` mismatches).

### Installation
```bash
python -m pip install -r requirements.txt
```

### Step 0: Acquire Raw Data (Skipped automatically if `data_raw/fdic/` is already populated)
```bash
python fdic_download_starter.py --start 1984-03-31 --end 2026-06-30 --out data_raw/fdic
```
*(Resumable: checks `data_raw/fdic/manifest_financials.json` and skips any quarter already downloaded and verified).*

### Steps 1–6: Execute the Complete Engineering, Modeling & Audit Pipeline
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

# Audit [Read-Only]: Export auxiliary table counts, reconciliations & supplementary DQ evidence tables
python src/06_export_evidence.py
```

---

## 4. Academic Integrity, References & AI-Tool Attribution (Section 4.3 Compliance)

In strict compliance with Section 4.3 of the BFT65B Final Examination guidelines (*"Cite every source you use (code, papers, AI tools) in the report and the README"*), all data sources, academic papers, Vietnamese regulatory frameworks, software packages, and AI tools used in this project are transparently documented below.

### 1. Primary Data Sources & Official Dictionaries
- **Federal Deposit Insurance Corporation (FDIC) — BankFind Suite Public REST API:** Call Report `/financials`, `/failures`, and `/institutions` endpoints (`https://api.fdic.gov/banks`; documentation at `https://api.fdic.gov/banks/docs/`). Downloaded via `fdic_download_starter.py` on **`2026-11-18`** (`manifest_financials.json` created `2026-11-18 23:29:02 +07:00` / `2026-11-18 16:29:02 UTC`; completeness verified in [`outputs/part1_manifest_summary.csv`](outputs/part1_manifest_summary.csv) and [`outputs/part1_aux_tables_summary.csv`](outputs/part1_aux_tables_summary.csv)).
- **FDIC RISView Field Dictionary:** Official Call Report field definitions and units (`https://api.fdic.gov/banks/docs/risview_properties.yaml`).
- **FDIC Regulatory & Resolution Documentation:** *Community Bank Leverage Ratio (CBLR) Framework (EGRRCPA Section 201, effective 2020Q1) & Call Report Schedule RC-R Instructions*; *FDIC Material Loss Reviews* (avoidable resolution cost basis for supervisory early intervention).

### 2. Academic Literature Cited
- **Cole, R. A., & White, L. J. (2012).** "Déjà Vu All Over Again: The Causes of U.S. Commercial Bank Failures This Time Around." *Journal of Financial Services Research*, 42(1), 5–29. *(Supervisory CAMELS proxies, commercial real-estate loan concentration, and early-intervention cost-savings context).*
- **Wheelock, D. C., & Wilson, P. W. (2000).** "Why Do Banks Disappear? The Determinants of U.S. Bank Failures and Acquisitions." *The Review of Economics and Statistics*, 82(1), 127–138. *(Econometric distinction between outright bank failures and non-failure mergers/acquisitions).*
- **Cassidy, G. (1980s, RBC Capital Markets).** Supervisory **Texas Ratio** formulation, `(NCLNLS + ORE) / (EQ + LNATRES)`, originally developed to identify insolvent institutions during the 1980s Texas S&L and commercial banking crisis.

### 3. Vietnam Banking Supervision & Regulatory Framework (SBV Adaptation)
- **National Assembly of Vietnam (2024).** *Law on Credit Institutions No. 32/2024/QH15* (Early Intervention *Can thiệp sớm*, Special Control *Kiểm soát đặc biệt*, and Compulsory Transfer *Chuyển giao bắt buộc*).
- **State Bank of Vietnam (SBV) (2016).** *Circular No. 41/2016/TT-NHNN* stipulating Basel II Capital Adequacy Ratio (CAR) standards for commercial banks and foreign bank branches.
- **State Bank of Vietnam (SBV) (2021).** *Circular No. 11/2021/TT-NHNN* governing asset classification (Loan Groups 1–5), risk provisioning, and use of credit loss reserves.
- **State Bank of Vietnam (SBV) (2020, 2023).** *Circular No. 01/2020/TT-NHNN* and *Circular No. 02/2023/TT-NHNN* on temporary loan debt-restructuring and forbearance classification.
- **Government of Vietnam & SBV (2013–present).** *Vietnam Asset Management Company (VAMC)* special-bond NPL transfer mechanism (*Decree No. 53/2013/ND-CP* and subsequent amendments), *National Credit Information Center of Vietnam (CIC)* borrower-level loan registry, and *CITAD / IBPS* interbank electronic payment settlement infrastructure.

### 4. Course Starter Code & Core Open-Source Software Libraries
- **Course Starter Code:** [`fdic_download_starter.py`](fdic_download_starter.py) provided as starter ingestion code by the BFT65B – Big Data Analytics in Finance instructors (National Economics University) and used unmodified for raw FDIC API acquisition.
- **Core Open-Source Libraries & Runtimes (Pinned in [`requirements.txt`](requirements.txt), verified in [`outputs/env_versions.txt`](outputs/env_versions.txt)):** Python `3.14.10`, Eclipse Temurin OpenJDK `17.0.21.1`, `polars==2.1.0` (Vink et al.), `pyarrow==26.0.1` (Apache Arrow), `pyspark==4.3.0` & `py4j==0.10.9.9` (Apache Spark), `scikit-learn==1.9.1` (Pedregosa et al., 2011), `pandas==3.1.1` (McKinney, 2010), `numpy==2.5.5` (Harris et al., 2020), `scipy==1.18.1`, `joblib==1.6.0`, `matplotlib==3.11.2`, `seaborn==0.13.2`, `psutil==7.2.2`, and `requests==2.34.2`.

### 5. Transparent AI-Tool Attribution
- **AI Coding & Research Assistant:** **Google Antigravity IDE (powered by Gemini)** was used to assist in structuring the modular Python pipeline (`src/01_storage_benchmark.py` through `src/06_export_evidence.py`), authoring automated data-quality and truncation zero-leakage assertions, verifying Polars-to-PySpark window equivalence syntax, and formatting tables, charts, and documentation.
- **Author Verification & Accountability:** All pipeline architecture decisions, point-in-time causal cleaning rules, temporal split and embargo choices, empirical benchmarks, and domain interpretations were independently executed, audited, and programmatically verified against the raw FDIC dataset and exported CSV/TXT evidence artifacts in `outputs/`.
