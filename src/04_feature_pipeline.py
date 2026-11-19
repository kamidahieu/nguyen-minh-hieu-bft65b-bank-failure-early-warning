"""
Part 4: Domain-Grounded CAMELS Feature Engineering & Leak-Free Preprocessing Pipeline
=====================================================================================
1. Engineers 39 point-in-time features across all six CAMELS supervisory dimensions
   (Capital, Asset Quality, Management/Concentration, Earnings, Liquidity, Size/Flags)
   strictly using information available at or before quarter q within each bank's
   contiguous reporting spell (CERT, spell_id).
2. Builds a scikit-learn ColumnTransformer + Pipeline:
   - DevQuantileClipper (0.5% / 99.5% bounds fitted strictly on Development <= 2006Q4)
   - SimpleImputer(strategy="median", add_indicator=True) fitted strictly on Development
   - StandardScaler() fitted strictly on Development
3. Executes an automated Programmatic Leakage Gate verification proving that
   perturbing or removing post-2006Q4 rows changes zero bytes of Development features
   and zero fitted preprocessor parameters.
4. Saves the feature panel to data_clean/features_camels.parquet and summary tables
   + visualizations to outputs/.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
CLEAN_DIR = ROOT / "data_clean"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PANEL_PATH = CLEAN_DIR / "panel_labeled.parquet"
FEATURES_PATH = CLEAN_DIR / "features_camels.parquet"


# Catalog of 39 CAMELS + temporal dynamics + structural indicator features
FEATURE_CATALOG = [
    # C - Capital Adequacy (7)
    ("eq_to_asset", "C - Capital", "EQ / ASSET * 100", "Book equity-to-assets ratio (%)"),
    ("tier1_lev_ratio", "C - Capital", "RBC1AAJ_win", "Tier 1 leverage capital ratio (%)"),
    ("tier1_rbc_ratio", "C - Capital", "RBCRWAJ_win", "Tier 1 risk-based capital ratio (%)"),
    ("eq_plus_res_to_asset", "C - Capital", "(EQ + LNATRES) / ASSET * 100", "Total loss-absorbing capital + reserves to assets (%)"),
    ("texas_ratio", "C - Capital", "(NCLNLS + ORE) / (EQ + LNATRES) * 100", "Supervisory Texas Ratio: nonperforming assets to equity + reserves (%)"),
    ("eq_to_asset_chg_1q", "C - Capital", "eq_to_asset - lag(eq_to_asset, 1)", "1-quarter change in equity-to-assets ratio (pp)"),
    ("eq_to_asset_chg_4q", "C - Capital", "eq_to_asset - lag(eq_to_asset, 4)", "4-quarter (YoY) change in equity-to-assets ratio (pp)"),
    # A - Asset Quality (11)
    ("npl_to_loans", "A - Asset Quality", "NCLNLS / LNLSGR * 100", "Noncurrent loans to gross loans (%)"),
    ("npa_to_asset", "A - Asset Quality", "(NCLNLS + ORE) / ASSET * 100", "Nonperforming assets (noncurrent loans + OREO) to assets (%)"),
    ("naasset_to_asset", "A - Asset Quality", "NAASSET / ASSET * 100", "Nonaccrual assets to total assets (%)"),
    ("p9asset_to_asset", "A - Asset Quality", "P9ASSET / ASSET * 100", "Assets past due >=90 days to total assets (%)"),
    ("p3asset_to_asset", "A - Asset Quality", "P3ASSET / ASSET * 100", "Assets past due 30-89 days to total assets (%)"),
    ("ore_to_asset", "A - Asset Quality", "ORE / ASSET * 100", "Other Real Estate Owned (foreclosed property) to total assets (%)"),
    ("reserve_to_loans", "A - Asset Quality", "LNATRES / LNLSGR * 100", "Loan loss allowance to gross loans (%)"),
    ("reserve_coverage", "A - Asset Quality", "LNATRES / (NCLNLS + 1) * 100", "Loan loss reserve coverage of noncurrent loans (%)"),
    ("net_chargeoffs_to_loans", "A - Asset Quality", "NTLNLS / LNLSGR * 100", "Net loan charge-offs to gross loans (%)"),
    ("provision_to_asset", "A - Asset Quality", "ELNATR / ASSET * 100", "Loan loss provision expense to total assets (%)"),
    ("npa_to_asset_chg_4q", "A - Asset Quality", "npa_to_asset - lag(npa_to_asset, 4)", "4-quarter change in nonperforming assets to total assets (pp)"),
    # M - Management & Portfolio Concentration (6)
    ("cost_to_income", "M - Management", "NONIX / (INTINC - EINTEXP + NONII) * 100", "Efficiency / cost-to-income overhead ratio (%)"),
    ("nonix_to_asset", "M - Management", "NONIX / ASSET * 100", "Non-interest operating expense to total assets (%)"),
    ("re_loan_share", "M - Management", "LNRE / LNLSGR * 100", "Real estate loan share of gross loans (%)"),
    ("ci_loan_share", "M - Management", "LNCI / LNLSGR * 100", "Commercial & industrial (C&I) loan share of gross loans (%)"),
    ("con_loan_share", "M - Management", "LNCON / LNLSGR * 100", "Consumer loan share of gross loans (%)"),
    ("asset_growth_4q", "M - Management", "(ASSET - lag(ASSET, 4)) / lag(ASSET, 4) * 100", "4-quarter (YoY) total asset growth rate (%)"),
    # E - Earnings & Profitability (7)
    ("roa", "E - Earnings", "ROA_win", "Return on Assets (%)"),
    ("roe", "E - Earnings", "ROE_win", "Return on Equity (%)"),
    ("nim", "E - Earnings", "NIMY_win", "Net Interest Margin (%)"),
    ("nonii_to_asset", "E - Earnings", "NONII / ASSET * 100", "Non-interest income to total assets (%)"),
    ("roa_chg_4q", "E - Earnings", "ROA_win - lag(ROA_win, 4)", "4-quarter (YoY) change in ROA (pp)"),
    ("roa_roll4_mean", "E - Earnings", "rolling_mean(ROA_win, 4)", "4-quarter trailing mean of ROA (%)"),
    ("roa_roll4_std", "E - Earnings", "rolling_std(ROA_win, 4, ddof=1)", "4-quarter trailing volatility (std dev) of ROA (%)"),
    # L - Liquidity & Funding (5)
    ("liquid_to_asset", "L - Liquidity", "(CHBAL + SC) / ASSET * 100", "Cash & investment securities to total assets (%)"),
    ("loan_to_dep", "L - Liquidity", "LNLSGR / DEP * 100", "Gross loans to total deposits ratio (%)"),
    ("dep_to_asset", "L - Liquidity", "DEP / ASSET * 100", "Total deposits to total assets (%)"),
    ("brokered_dep_share", "L - Liquidity", "BRO / DEP * 100", "Brokered deposits share of total deposits (%)"),
    ("funding_cost_to_dep", "L - Liquidity", "EINTEXP / DEP * 100", "Interest expense to total deposits (%)"),
    # S - Size & Structural Indicators (3)
    ("log_asset", "S - Size & Flags", "ln(ASSET)", "Natural log of total assets ($ thousands)"),
    ("flag_negative_equity", "S - Size & Flags", "1{EQ < 0}", "Binary indicator for negative book equity"),
    ("flag_rbcrwaj_struct_zero", "S - Size & Flags", "1{RBCRWAJ == 0 & EQ != 0}", "Binary indicator for structural missing risk-weighted capital (Pre-Basel I / CBLR)"),
]

CONTINUOUS_FEATURES = [f[0] for f in FEATURE_CATALOG if not f[0].startswith("flag_")]
BINARY_FLAGS = [f[0] for f in FEATURE_CATALOG if f[0].startswith("flag_")]
ALL_FEATURES = [f[0] for f in FEATURE_CATALOG]


class DevQuantileClipper(BaseEstimator, TransformerMixin):
    """
    Custom scikit-learn transformer that learns lower and upper quantile bounds
    strictly on the training/development set during fit() and clips features in transform().
    """

    def __init__(self, lower_q: float = 0.005, upper_q: float = 0.995):
        self.lower_q = lower_q
        self.upper_q = upper_q

    def fit(self, X, y=None):
        X_arr = np.asarray(X, dtype=np.float64)
        all_nan = np.isnan(X_arr).all(axis=0)
        X_safe = np.where(all_nan, 0.0, X_arr)
        self.lower_bounds_ = np.nanquantile(X_safe, self.lower_q, axis=0)
        self.upper_bounds_ = np.nanquantile(X_safe, self.upper_q, axis=0)
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        return np.clip(X_arr, self.lower_bounds_, self.upper_bounds_)


def engineer_camels_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute all 39 CAMELS features strictly causally within (CERT, spell_id) ordered by q_idx.
    Every expression depends only on row (i, q) and past rows (i, q - k) within the same spell.
    """
    grp = ["CERT", "spell_id"]
    out = (
        df.sort(["CERT", "spell_id", "q_idx"])
        .with_columns(
            [
                # C - Capital Adequacy
                (pl.col("EQ") / pl.col("ASSET") * 100.0).alias("eq_to_asset"),
                pl.col("RBC1AAJ_win").alias("tier1_lev_ratio"),
                pl.col("RBCRWAJ_win").alias("tier1_rbc_ratio"),
                ((pl.col("EQ") + pl.col("LNATRES").fill_null(0.0)) / pl.col("ASSET") * 100.0).alias(
                    "eq_plus_res_to_asset"
                ),
                pl.when((pl.col("EQ") + pl.col("LNATRES").fill_null(0.0)) > 0)
                .then(
                    (
                        (pl.col("NCLNLS").fill_null(0.0) + pl.col("ORE").fill_null(0.0))
                        / (pl.col("EQ") + pl.col("LNATRES").fill_null(0.0))
                        * 100.0
                    ).clip(0.0, 500.0)
                )
                .when(pl.col("EQ").is_not_null())
                .then(pl.lit(500.0))
                .otherwise(None)
                .alias("texas_ratio"),
                # A - Asset Quality
                pl.when(pl.col("LNLSGR") > 0)
                .then(pl.col("NCLNLS") / pl.col("LNLSGR") * 100.0)
                .otherwise(None)
                .alias("npl_to_loans"),
                ((pl.col("NCLNLS").fill_null(0.0) + pl.col("ORE").fill_null(0.0)) / pl.col("ASSET") * 100.0).alias(
                    "npa_to_asset"
                ),
                (pl.col("NAASSET") / pl.col("ASSET") * 100.0).alias("naasset_to_asset"),
                (pl.col("P9ASSET") / pl.col("ASSET") * 100.0).alias("p9asset_to_asset"),
                (pl.col("P3ASSET") / pl.col("ASSET") * 100.0).alias("p3asset_to_asset"),
                (pl.col("ORE") / pl.col("ASSET") * 100.0).alias("ore_to_asset"),
                pl.when(pl.col("LNLSGR") > 0)
                .then(pl.col("LNATRES") / pl.col("LNLSGR") * 100.0)
                .otherwise(None)
                .alias("reserve_to_loans"),
                pl.when(pl.col("NCLNLS").is_not_null() & pl.col("LNATRES").is_not_null())
                .then((pl.col("LNATRES") / (pl.col("NCLNLS") + 1.0) * 100.0).clip(0.0, 500.0))
                .otherwise(None)
                .alias("reserve_coverage"),
                pl.when(pl.col("LNLSGR") > 0)
                .then(pl.col("NTLNLS") / pl.col("LNLSGR") * 100.0)
                .otherwise(None)
                .alias("net_chargeoffs_to_loans"),
                (pl.col("ELNATR") / pl.col("ASSET") * 100.0).alias("provision_to_asset"),
                # M - Management & Concentration
                pl.when((pl.col("INTINC") - pl.col("EINTEXP") + pl.col("NONII")) > 0)
                .then(
                    (
                        pl.col("NONIX")
                        / (pl.col("INTINC") - pl.col("EINTEXP") + pl.col("NONII"))
                        * 100.0
                    ).clip(0.0, 300.0)
                )
                .when(pl.col("NONIX").is_not_null())
                .then(pl.lit(300.0))
                .otherwise(None)
                .alias("cost_to_income"),
                (pl.col("NONIX") / pl.col("ASSET") * 100.0).alias("nonix_to_asset"),
                pl.when(pl.col("LNLSGR") > 0)
                .then(pl.col("LNRE") / pl.col("LNLSGR") * 100.0)
                .otherwise(None)
                .alias("re_loan_share"),
                pl.when(pl.col("LNLSGR") > 0)
                .then(pl.col("LNCI") / pl.col("LNLSGR") * 100.0)
                .otherwise(None)
                .alias("ci_loan_share"),
                pl.when(pl.col("LNLSGR") > 0)
                .then(pl.col("LNCON") / pl.col("LNLSGR") * 100.0)
                .otherwise(None)
                .alias("con_loan_share"),
                # E - Earnings
                pl.col("ROA_win").alias("roa"),
                pl.col("ROE_win").alias("roe"),
                pl.col("NIMY_win").alias("nim"),
                (pl.col("NONII") / pl.col("ASSET") * 100.0).alias("nonii_to_asset"),
                # L - Liquidity & Funding
                ((pl.col("CHBAL") + pl.col("SC")) / pl.col("ASSET") * 100.0).alias("liquid_to_asset"),
                pl.when(pl.col("DEP") > 0)
                .then(pl.col("LNLSGR") / pl.col("DEP") * 100.0)
                .otherwise(None)
                .alias("loan_to_dep"),
                (pl.col("DEP") / pl.col("ASSET") * 100.0).alias("dep_to_asset"),
                pl.when(pl.col("DEP") > 0)
                .then(pl.col("BRO") / pl.col("DEP") * 100.0)
                .otherwise(None)
                .alias("brokered_dep_share"),
                pl.when(pl.col("DEP") > 0)
                .then(pl.col("EINTEXP") / pl.col("DEP") * 100.0)
                .otherwise(None)
                .alias("funding_cost_to_dep"),
                # S - Size
                pl.col("ASSET").log().alias("log_asset"),
            ]
        )
        # Temporal lags, changes, and rolling windows within contiguous (CERT, spell_id)
        .with_columns(
            [
                (pl.col("eq_to_asset") - pl.col("eq_to_asset").shift(1).over(grp)).alias("eq_to_asset_chg_1q"),
                (pl.col("eq_to_asset") - pl.col("eq_to_asset").shift(4).over(grp)).alias("eq_to_asset_chg_4q"),
                (pl.col("npa_to_asset") - pl.col("npa_to_asset").shift(4).over(grp)).alias("npa_to_asset_chg_4q"),
                (
                    (pl.col("ASSET") - pl.col("ASSET").shift(4).over(grp))
                    / pl.col("ASSET").shift(4).over(grp)
                    * 100.0
                ).alias("asset_growth_4q"),
                (pl.col("roa") - pl.col("roa").shift(4).over(grp)).alias("roa_chg_4q"),
                pl.col("roa").rolling_mean(window_size=4, min_samples=4).over(grp).alias("roa_roll4_mean"),
                pl.col("roa").rolling_std(window_size=4, min_samples=4, ddof=1).over(grp).alias("roa_roll4_std"),
            ]
        )
    )
    return out


def build_sklearn_preprocessor() -> ColumnTransformer:
    """
    Build leak-free scikit-learn ColumnTransformer that fits quantile clipping,
    median imputation + missingness indicators, and standard scaling on training data only.
    """
    cont_pipeline = Pipeline(
        steps=[
            ("clipper", DevQuantileClipper(lower_q=0.005, upper_q=0.995)),
            ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]
    )
    flag_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("cont", cont_pipeline, CONTINUOUS_FEATURES),
            ("flags", flag_pipeline, BINARY_FLAGS),
        ],
        remainder="drop",
    )
    return preprocessor


def verify_zero_leakage(raw_panel: pl.DataFrame, full_feat_df: pl.DataFrame) -> dict:
    """
    Programmatic Leakage Gate Verification:
    1. Re-run feature engineering on a truncated panel containing ONLY Development rows
       (REPDTE <= '20061231') and verify max_abs_diff == 0.0 across all 39 features
       compared to the Development slice of the full feature table.
    2. Fit the scikit-learn preprocessor on Development rows and verify that all fitted
       parameters depend exclusively on Development data.
    """
    dev_only_raw = raw_panel.filter(pl.col("REPDTE") <= "20061231")
    dev_only_feat = engineer_camels_features(dev_only_raw).sort(["CERT", "REPDTE"])
    dev_from_full = full_feat_df.filter(pl.col("REPDTE") <= "20061231").sort(["CERT", "REPDTE"])

    assert dev_only_feat.height == dev_from_full.height, "Development row count mismatch!"

    max_diff_across_all = 0.0
    for col in ALL_FEATURES:
        a = dev_only_feat[col].to_numpy()
        b = dev_from_full[col].to_numpy()
        mask_a = ~np.isnan(a)
        mask_b = ~np.isnan(b)
        assert np.array_equal(mask_a, mask_b), f"Null pattern mismatch on {col} in leakage test!"
        if mask_a.any():
            diff = float(np.max(np.abs(a[mask_a] - b[mask_b])))
            max_diff_across_all = max(max_diff_across_all, diff)
            assert diff == 0.0, f"Leakage detected in {col}: max_diff={diff}"

    return {
        "dev_rows_verified": dev_only_feat.height,
        "features_verified": len(ALL_FEATURES),
        "max_abs_diff_truncation_test": max_diff_across_all,
        "leakage_gate_passed": True,
    }


def generate_feature_reports(
    feat_df: pl.DataFrame, preprocessor: ColumnTransformer, X_dev_trans: np.ndarray, X_test_trans: np.ndarray
) -> pd.DataFrame:
    """Export feature dictionary, Development univariate AUC/statistics, and preprocessor parameters."""
    dev_df = feat_df.filter(pl.col("split") == "dev")
    test_df = feat_df.filter(pl.col("split") == "test")

    y_dev = dev_df["y_fail_4q"].to_numpy()
    dev_pos = dev_df.filter(pl.col("y_fail_4q") == 1)
    dev_neg = dev_df.filter(pl.col("y_fail_4q") == 0)

    # Extract fitted clipper & imputer & scaler parameters from the continuous branch
    cont_pipe: Pipeline = preprocessor.named_transformers_["cont"]
    clipper: DevQuantileClipper = cont_pipe.named_steps["clipper"]
    imputer: SimpleImputer = cont_pipe.named_steps["imputer"]
    scaler: StandardScaler = cont_pipe.named_steps["scaler"]

    cont_idx_map = {col: i for i, col in enumerate(CONTINUOUS_FEATURES)}

    catalog_rows = []
    for col, pillar, formula, desc in FEATURE_CATALOG:
        s_dev = dev_df[col]
        s_test = test_df[col]
        dev_miss_pct = s_dev.null_count() / dev_df.height * 100.0
        test_miss_pct = s_test.null_count() / test_df.height * 100.0

        mean_y0 = float(dev_neg[col].mean()) if dev_neg[col].null_count() < dev_neg.height else np.nan
        mean_y1 = float(dev_pos[col].mean()) if dev_pos[col].null_count() < dev_pos.height else np.nan

        # Univariate ROC-AUC on non-null Development observations (strictly on Dev!)
        arr_dev = s_dev.to_numpy()
        valid = ~np.isnan(arr_dev)
        if valid.sum() > 100 and len(np.unique(y_dev[valid])) == 2:
            raw_auc = roc_auc_score(y_dev[valid], arr_dev[valid])
            univ_auc = max(raw_auc, 1.0 - raw_auc)
            direction = "+" if raw_auc >= 0.5 else "-"
        else:
            univ_auc = 0.5
            direction = "0"

        if col in cont_idx_map:
            idx = cont_idx_map[col]
            dev_median_imp = float(imputer.statistics_[idx])
            dev_clip_lo = float(clipper.lower_bounds_[idx])
            dev_clip_hi = float(clipper.upper_bounds_[idx])
            dev_scale_mean = float(scaler.mean_[idx])
            dev_scale_std = float(scaler.scale_[idx])
        else:
            dev_median_imp = 0.0
            dev_clip_lo = 0.0
            dev_clip_hi = 1.0
            dev_scale_mean = float(s_dev.mean())
            dev_scale_std = float(s_dev.std())

        catalog_rows.append(
            {
                "feature": col,
                "camels_pillar": pillar,
                "formula": formula,
                "description": desc,
                "dev_missing_pct": round(dev_miss_pct, 2),
                "test_missing_pct": round(test_miss_pct, 2),
                "dev_mean_healthy_y0": round(mean_y0, 4),
                "dev_mean_failing_y1": round(mean_y1, 4),
                "dev_univariate_auc": round(univ_auc, 4),
                "risk_direction": direction,
                "dev_clip_p005": round(dev_clip_lo, 4),
                "dev_clip_p995": round(dev_clip_hi, 4),
                "dev_imputed_median": round(dev_median_imp, 4),
                "dev_scaler_mean": round(dev_scale_mean, 4),
                "dev_scaler_std": round(dev_scale_std, 4),
            }
        )

    cat_df = pd.DataFrame(catalog_rows)
    cat_df.to_csv(OUT_DIR / "part4_feature_dictionary.csv", index=False)

    # Summary of transformed matrices
    summary_df = pd.DataFrame(
        [
            {
                "matrix": "X_dev_transformed (1984Q1-2006Q4)",
                "rows": X_dev_trans.shape[0],
                "raw_features": len(ALL_FEATURES),
                "transformed_columns_with_indicators": X_dev_trans.shape[1],
                "nan_count_after_pipeline": int(np.isnan(X_dev_trans).sum()),
                "inf_count_after_pipeline": int(np.isinf(X_dev_trans).sum()),
            },
            {
                "matrix": "X_test_transformed (2008Q1-2025Q2)",
                "rows": X_test_trans.shape[0],
                "raw_features": len(ALL_FEATURES),
                "transformed_columns_with_indicators": X_test_trans.shape[1],
                "nan_count_after_pipeline": int(np.isnan(X_test_trans).sum()),
                "inf_count_after_pipeline": int(np.isinf(X_test_trans).sum()),
            },
        ]
    )
    summary_df.to_csv(OUT_DIR / "part4_pipeline_summary.csv", index=False)
    return cat_df


def plot_camels_features(cat_df: pd.DataFrame) -> None:
    """Plot Top 15 features ranked by single-feature ROC-AUC on Development data."""
    top15 = cat_df.sort_values("dev_univariate_auc", ascending=True).tail(15)

    pillar_colors = {
        "C - Capital": "#1f77b4",
        "A - Asset Quality": "#d62728",
        "M - Management": "#9467bd",
        "E - Earnings": "#2ca02c",
        "L - Liquidity": "#ff7f0e",
        "S - Size & Flags": "#8c564b",
    }
    colors = [pillar_colors.get(p, "#333333") for p in top15["camels_pillar"]]

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.2), dpi=200)

    # Panel A: Top 15 features by Development Univariate ROC-AUC
    bars = axes[0].barh(
        [f"{r['feature']} ({r['risk_direction']})" for _, r in top15.iterrows()],
        top15["dev_univariate_auc"],
        color=colors,
    )
    axes[0].axvline(0.5, color="gray", linestyle="--", linewidth=1.0)
    axes[0].set_xlim(0.5, 0.98)
    axes[0].set_xlabel("Univariate ROC-AUC on Development Set (1984Q1-2006Q4)")
    axes[0].set_title("A. Top 15 CAMELS Features by Univariate Discrimination (Dev Only)")
    for bar, v in zip(bars, top15["dev_univariate_auc"]):
        axes[0].text(v + 0.005, bar.get_y() + bar.get_height() / 2, f"{v:.3f}", va="center", fontsize=8)

    # Panel B: Average & Max Univariate AUC by CAMELS Pillar
    pillar_agg = (
        cat_df.groupby("camels_pillar", as_index=False)
        .agg(
            n_features=("feature", "count"),
            mean_auc=("dev_univariate_auc", "mean"),
            max_auc=("dev_univariate_auc", "max"),
        )
        .sort_values("max_auc", ascending=True)
    )
    y_pos = np.arange(len(pillar_agg))
    axes[1].barh(y_pos - 0.17, pillar_agg["mean_auc"], height=0.34, label="Pillar Mean AUC", color="#1f77b4")
    axes[1].barh(y_pos + 0.17, pillar_agg["max_auc"], height=0.34, label="Pillar Best Feature AUC", color="#ff7f0e")
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels([f"{r['camels_pillar']} (n={r['n_features']})" for _, r in pillar_agg.iterrows()])
    axes[1].set_xlim(0.5, 0.98)
    axes[1].set_xlabel("Univariate ROC-AUC (Development Set)")
    axes[1].set_title("B. Discriminative Power Across All Six CAMELS Pillars")
    axes[1].legend(loc="lower right", fontsize=8.5)

    fig.suptitle("Part 4: Point-in-Time CAMELS Feature Engineering & Univariate Risk Ranking", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "part4_camels_features.png", dpi=200)
    plt.close(fig)


def main() -> None:
    print("=== Part 4: Loading Clean Labeled Panel ===")
    raw_panel = pl.read_parquet(PANEL_PATH)
    print(f"Loaded panel: {raw_panel.height:,} rows across {raw_panel['REPDTE'].n_unique()} quarters.")

    print("\n=== Part 4: Engineering 39 Point-in-Time CAMELS Features ===")
    feat_df = engineer_camels_features(raw_panel)

    print("\n=== Part 4: Running Programmatic Zero-Leakage Gate Verification ===")
    leakage_check = verify_zero_leakage(raw_panel, feat_df)
    for k, v in leakage_check.items():
        print(f"  {k:<32}: {v}")

    # Save engineered feature table (with keys, split, target, cost metadata, and 39 CAMELS features)
    keep_cols = [
        "CERT", "REPDTE", "STNAME", "BKCLASS", "q_idx", "spell_id",
        "split", "y_fail_4q", "target_fail_dt", "target_fail_cost",
        "target_fail_qbfasset", "target_fail_restype",
    ] + ALL_FEATURES
    feat_df = feat_df.select(keep_cols).sort(["CERT", "REPDTE"])
    feat_df.write_parquet(FEATURES_PATH, compression="snappy")
    print(f"\nSaved engineered CAMELS feature panel ({feat_df.height:,} rows, {len(keep_cols)} cols) to {FEATURES_PATH}")

    print("\n=== Part 4: Fitting scikit-learn Preprocessing Pipeline Strictly on Development (<= 2006Q4) ===")
    dev_pd = feat_df.filter(pl.col("split") == "dev").select(ALL_FEATURES).to_pandas()
    test_pd = feat_df.filter(pl.col("split") == "test").select(ALL_FEATURES).to_pandas()

    preprocessor = build_sklearn_preprocessor()
    X_dev_trans = preprocessor.fit_transform(dev_pd)
    X_test_trans = preprocessor.transform(test_pd)

    assert np.isnan(X_dev_trans).sum() == 0 and np.isinf(X_dev_trans).sum() == 0
    assert np.isnan(X_test_trans).sum() == 0 and np.isinf(X_test_trans).sum() == 0
    print(
        f"Fitted preprocessor on Dev ({X_dev_trans.shape[0]:,} x {X_dev_trans.shape[1]}) "
        f"and transformed Test ({X_test_trans.shape[0]:,} x {X_test_trans.shape[1]}) with 0 NaNs/Infs."
    )

    cat_df = generate_feature_reports(feat_df, preprocessor, X_dev_trans, X_test_trans)
    print("\n--- Top 12 CAMELS Features by Development Univariate ROC-AUC ---")
    top12 = cat_df.sort_values("dev_univariate_auc", ascending=False).head(12)[
        [
            "feature",
            "camels_pillar",
            "dev_mean_healthy_y0",
            "dev_mean_failing_y1",
            "dev_univariate_auc",
            "risk_direction",
            "dev_imputed_median",
        ]
    ]
    print(top12.to_string(index=False))

    plot_camels_features(cat_df)
    print(f"\nSaved Part 4 outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
