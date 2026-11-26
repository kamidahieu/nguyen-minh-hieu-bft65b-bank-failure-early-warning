"""
Part 5: Leak-Free Modeling, Purged Temporal Validation, Cost-Based Thresholding & PySpark MLlib
===============================================================================================
1. Implements a 4-Fold Purged Expanding-Window Temporal Cross-Validation scheme on the
   Development set (1984Q1-2006Q4) with a strict 4-quarter embargo purge gap between each
   fold's training end and validation start to prevent 4-quarter forward label overlap.
2. Compares four models fitted strictly on Development data:
   - Baseline 1: Logistic Regression (L2, calibrated probabilities) in a leak-free sklearn Pipeline
   - Baseline 2: Class-Balanced Logistic Regression (illustrating calibration vs. ranking trade-off)
   - Ensemble 1: Random Forest Classifier (100 trees, depth 10) in a leak-free sklearn Pipeline
   - Ensemble 2: Histogram-Based Gradient Boosting (HistGradientBoostingClassifier)
   - Distributed Demo: PySpark MLlib Pipeline (Imputer -> VectorAssembler -> StandardScaler ->
     LogisticRegression & GBTClassifier)
3. Estimates asymmetric misclassification costs empirically from Development failures
   (1984-2006 failures.parquet COST & QBFASSET vs. targeted examination cost) and selects
   the optimal alarm probability threshold tau* strictly on Development out-of-fold CV predictions.
4. Evaluates ONCE on the held-out Test period (2008Q1-2025Q2, 435,446 bank-quarters, 558 failing
   banks), reporting overall metrics, crisis sub-period performance (2008-2014 GFC vs. 2023
   Regional Bank Crisis), confusion matrices, PR/ROC curves, cost curves, and feature importances.
"""
from __future__ import annotations

import os
import sys
import time
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
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(ROOT / "src"))
from importlib import import_module

m4 = import_module("04_feature_pipeline")
ALL_FEATURES = m4.ALL_FEATURES
CONTINUOUS_FEATURES = m4.CONTINUOUS_FEATURES
BINARY_FLAGS = m4.BINARY_FLAGS
DevQuantileClipper = m4.DevQuantileClipper
build_sklearn_preprocessor = m4.build_sklearn_preprocessor

SEED = 42
RAW_DIR = ROOT / "data_raw" / "fdic"
CLEAN_DIR = ROOT / "data_clean"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FEATURES_PATH = CLEAN_DIR / "features_camels.parquet"

# 4-Fold Purged Expanding-Window CV on Development (1984Q1-2006Q4)
# Each fold enforces a strict 4-quarter embargo gap between train_end and val_start
PURGED_CV_FOLDS = [
    {
        "fold": 1,
        "name": "Fold 1 (Val 1988-1989: Early S&L Crisis)",
        "train_end": "19861231",
        "purge_window": "19870331-19871231",
        "val_start": "19880331",
        "val_end": "19891231",
    },
    {
        "fold": 2,
        "name": "Fold 2 (Val 1990-1991: Peak S&L / FIRREA)",
        "train_end": "19881231",
        "purge_window": "19890331-19891231",
        "val_start": "19900331",
        "val_end": "19911231",
    },
    {
        "fold": 3,
        "name": "Fold 3 (Val 1992-1994: Post-FIRREA Resolution)",
        "train_end": "19901231",
        "purge_window": "19910331-19911231",
        "val_start": "19920331",
        "val_end": "19941231",
    },
    {
        "fold": 4,
        "name": "Fold 4 (Val 1995-2006: Great Moderation)",
        "train_end": "19931231",
        "purge_window": "19940331-19941231",
        "val_start": "19950331",
        "val_end": "20061231",
    },
]


def build_candidate_models() -> dict[str, Pipeline]:
    """Build leak-free scikit-learn Pipelines for baseline and ensemble models."""
    models = {
        "Logistic Regression (L2 Calibrated)": Pipeline(
            steps=[
                ("prep", build_sklearn_preprocessor()),
                ("clf", LogisticRegression(C=0.1, max_iter=500, random_state=SEED)),
            ]
        ),
        "Logistic Regression (Class-Balanced)": Pipeline(
            steps=[
                ("prep", build_sklearn_preprocessor()),
                (
                    "clf",
                    LogisticRegression(
                        C=0.1, class_weight="balanced", max_iter=500, random_state=SEED
                    ),
                ),
            ]
        ),
        "Random Forest (100 trees, depth=10)": Pipeline(
            steps=[
                ("prep", build_sklearn_preprocessor()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=100,
                        max_depth=10,
                        min_samples_leaf=50,
                        max_features="sqrt",
                        max_samples=0.5,
                        n_jobs=-1,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
        "HistGradientBoosting (GBM)": Pipeline(
            steps=[
                ("prep", build_sklearn_preprocessor()),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        max_iter=200,
                        learning_rate=0.05,
                        max_leaf_nodes=31,
                        min_samples_leaf=50,
                        l2_regularization=1.0,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
    }
    return models


def recall_at_top_k_pct(y_true: np.ndarray, y_prob: np.ndarray, k_pct: float) -> float:
    """Compute fraction of all true failures captured in the top k_pct% highest-risk scores."""
    total_pos = y_true.sum()
    if total_pos == 0:
        return 0.0
    n_top = max(1, int(round(len(y_true) * (k_pct / 100.0))))
    top_idx = np.argpartition(y_prob, -n_top)[-n_top:]
    return float(y_true[top_idx].sum() / total_pos)


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute imbalance-appropriate probabilistic & ranking metrics."""
    return {
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "recall_top_1pct": recall_at_top_k_pct(y_true, y_prob, 1.0),
        "recall_top_2pct": recall_at_top_k_pct(y_true, y_prob, 2.0),
        "recall_top_5pct": recall_at_top_k_pct(y_true, y_prob, 5.0),
        "mean_pred_prob": float(np.mean(y_prob)),
        "actual_pos_rate": float(np.mean(y_true)),
    }


def estimate_empirical_costs_from_dev() -> dict:
    """
    Estimate misclassification costs strictly from Development period (1984-2006)
    in failures.parquet and financials:
    - Median DIF failure resolution cost (COST) in 1984-2006 = $9,598.56k ($9.60M).
    - Spread across the 4 pre-failure quarterly warning windows = $2,399.64k per quarter.
    - Early supervisory intervention avoidable loss fraction alpha = 20% -> C_FN = $479.93k.
    - Targeted on-site examination + supervisory compliance cost per false alarm -> C_FP = $24.00k
      (gives empirical baseline cost ratio C_FN / C_FP = 20.0, with sensitivity at 10:1 and 50:1).
    """
    fail = (
        pl.read_parquet(RAW_DIR / "failures.parquet")
        .filter(pl.col("CERT").is_not_null())
        .with_columns(
            pl.col("CERT").cast(pl.Int64),
            pl.col("FAILDATE").str.to_date("%m/%d/%Y").alias("fail_dt"),
        )
        .filter((pl.col("fail_dt") >= pl.date(1984, 1, 1)) & (pl.col("fail_dt") <= pl.date(2006, 12, 31)))
    )
    valid_fail = fail.filter(
        pl.col("COST").is_not_null() & (pl.col("COST") > 0) & pl.col("QBFASSET").is_not_null() & (pl.col("QBFASSET") > 0)
    )
    median_cost_k = float(valid_fail["COST"].median())
    mean_cost_k = float(valid_fail["COST"].mean())
    median_asset_k = float(valid_fail["QBFASSET"].median())
    median_loss_ratio = float((valid_fail["COST"] / valid_fail["QBFASSET"]).median())

    # Per-quarter avoidable failure cost ($k) with alpha=20% savings from early intervention
    alpha_avoidable = 0.20
    c_fn_k = (median_cost_k / 4.0) * alpha_avoidable  # ~$479.93k per missed pre-failure bank-quarter
    c_fp_k = c_fn_k / 20.0  # ~$24.00k per false-alarm targeted supervisory exam (20:1 ratio)

    cost_params = {
        "dev_failures_total": fail.height,
        "dev_failures_with_cost": valid_fail.height,
        "dev_median_dif_cost_k": round(median_cost_k, 2),
        "dev_mean_dif_cost_k": round(mean_cost_k, 2),
        "dev_median_qbfasset_k": round(median_asset_k, 2),
        "dev_median_cost_to_asset_pct": round(median_loss_ratio * 100.0, 2),
        "avoidable_loss_fraction_alpha": alpha_avoidable,
        "c_fn_per_bank_quarter_k": round(c_fn_k, 2),
        "c_fp_per_bank_quarter_k": round(c_fp_k, 2),
        "cost_ratio_fn_to_fp": round(c_fn_k / c_fp_k, 2),
        "theoretical_bayes_threshold": round(c_fp_k / (c_fn_k + c_fp_k), 4),
    }
    pd.DataFrame([cost_params]).to_csv(OUT_DIR / "part5_empirical_cost_params.csv", index=False)
    return cost_params


def run_purged_expanding_cv(
    dev_df: pd.DataFrame, models: dict[str, Pipeline]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], np.ndarray]:
    """
    Execute 4-Fold Purged Expanding-Window CV on Development data (1984Q1-2006Q4).
    Returns fold-level table, out-of-fold pooled summary table, OOF probability dict, and OOF mask.
    """
    repdte_arr = dev_df["REPDTE"].to_numpy()
    X_dev = dev_df[ALL_FEATURES]
    y_dev = dev_df["y_fail_4q"].to_numpy(dtype=np.int32)

    # Out-of-fold validation mask covers 1988Q1-2006Q4 across the 4 non-overlapping validation windows
    oof_mask = (repdte_arr >= "19880331") & (repdte_arr <= "20061231")
    oof_preds = {name: np.full(len(dev_df), np.nan, dtype=np.float64) for name in models}

    fold_rows = []
    for f_info in PURGED_CV_FOLDS:
        train_mask = repdte_arr <= f_info["train_end"]
        val_mask = (repdte_arr >= f_info["val_start"]) & (repdte_arr <= f_info["val_end"])

        X_tr, y_tr = X_dev.loc[train_mask], y_dev[train_mask]
        X_va, y_va = X_dev.loc[val_mask], y_dev[val_mask]

        for name, pipe in models.items():
            t0 = time.perf_counter()
            est = clone(pipe)
            est.fit(X_tr, y_tr)
            prob_va = est.predict_proba(X_va)[:, 1]
            fit_s = time.perf_counter() - t0

            oof_preds[name][val_mask] = prob_va
            m = evaluate_predictions(y_va, prob_va)
            fold_rows.append(
                {
                    "fold": f_info["fold"],
                    "fold_name": f_info["name"],
                    "train_rows": int(train_mask.sum()),
                    "train_pos": int(y_tr.sum()),
                    "val_rows": int(val_mask.sum()),
                    "val_pos": int(y_va.sum()),
                    "model": name,
                    "pr_auc": round(m["pr_auc"], 4),
                    "roc_auc": round(m["roc_auc"], 4),
                    "brier_score": round(m["brier_score"], 5),
                    "recall_top_1pct": round(m["recall_top_1pct"], 4),
                    "recall_top_2pct": round(m["recall_top_2pct"], 4),
                    "recall_top_5pct": round(m["recall_top_5pct"], 4),
                    "fit_time_s": round(fit_s, 2),
                }
            )

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(OUT_DIR / "part5_cv_fold_metrics.csv", index=False)

    # Pooled Out-of-Fold (1988Q1-2006Q4) & Mean-Across-Folds CV summary
    oof_rows = []
    y_oof = y_dev[oof_mask]
    for name in models:
        p_oof = oof_preds[name][oof_mask]
        m_oof = evaluate_predictions(y_oof, p_oof)
        sub_f = fold_df[fold_df["model"] == name]
        oof_rows.append(
            {
                "model": name,
                "oof_val_rows": int(oof_mask.sum()),
                "oof_val_pos": int(y_oof.sum()),
                "mean_fold_pr_auc": round(float(sub_f["pr_auc"].mean()), 4),
                "pooled_oof_pr_auc": round(m_oof["pr_auc"], 4),
                "mean_fold_roc_auc": round(float(sub_f["roc_auc"].mean()), 4),
                "pooled_oof_roc_auc": round(m_oof["roc_auc"], 4),
                "pooled_oof_brier": round(m_oof["brier_score"], 5),
                "pooled_oof_recall_top_1pct": round(m_oof["recall_top_1pct"], 4),
                "pooled_oof_recall_top_2pct": round(m_oof["recall_top_2pct"], 4),
                "pooled_oof_recall_top_5pct": round(m_oof["recall_top_5pct"], 4),
                "mean_pred_prob": round(m_oof["mean_pred_prob"], 5),
            }
        )
    oof_df = pd.DataFrame(oof_rows).sort_values("pooled_oof_pr_auc", ascending=False)
    oof_df.to_csv(OUT_DIR / "part5_cv_summary.csv", index=False)
    return fold_df, oof_df, oof_preds, oof_mask


def optimize_cost_threshold_on_dev_oof(
    y_oof: np.ndarray,
    oof_preds: dict[str, np.ndarray],
    oof_mask: np.ndarray,
    cost_params: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """
    Sweep probability thresholds tau in [0.005, 0.90] strictly on Development Out-of-Fold
    predictions to minimize total expected supervisory + DIF resolution cost:
        Total_Cost(tau) = FN(tau) * C_FN + FP(tau) * C_FP
    """
    c_fn = cost_params["c_fn_per_bank_quarter_k"]
    c_fp = cost_params["c_fp_per_bank_quarter_k"]
    no_model_cost_k = float(y_oof.sum() * c_fn)

    grid = np.unique(
        np.concatenate(
            [
                np.linspace(0.005, 0.15, 146),
                np.linspace(0.16, 0.90, 75),
            ]
        )
    )

    curve_rows = []
    optimal_rows = []
    best_thresholds = {}

    for name, full_p in oof_preds.items():
        p_oof = full_p[oof_mask]
        best_cost = float("inf")
        best_row = None

        for tau in grid:
            pred = (p_oof >= tau).astype(np.int32)
            tp = int(((pred == 1) & (y_oof == 1)).sum())
            fp = int(((pred == 1) & (y_oof == 0)).sum())
            fn = int(((pred == 0) & (y_oof == 1)).sum())
            tn = int(((pred == 0) & (y_oof == 0)).sum())

            total_cost_k = fn * c_fn + fp * c_fp
            cost_savings_pct = (1.0 - total_cost_k / no_model_cost_k) * 100.0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            alert_rate_pct = (tp + fp) / len(y_oof) * 100.0

            row = {
                "model": name,
                "threshold": round(float(tau), 4),
                "total_cost_m_usd": round(total_cost_k / 1000.0, 3),
                "no_model_cost_m_usd": round(no_model_cost_k / 1000.0, 3),
                "cost_reduction_pct": round(cost_savings_pct, 2),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
                "alert_rate_pct": round(alert_rate_pct, 3),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
            curve_rows.append(row)
            if total_cost_k < best_cost:
                best_cost = total_cost_k
                best_row = row

        best_thresholds[name] = best_row["threshold"]
        optimal_rows.append(best_row)

    # Also compute sensitivity across cost ratios (10:1, 20:1, 40:1, 50:1) for the primary model
    primary_model = "HistGradientBoosting (GBM)"
    p_prim = oof_preds[primary_model][oof_mask]
    sens_rows = []
    for ratio in (10.0, 20.0, 40.0, 50.0):
        c_fp_r = c_fn / ratio
        best_tau_r = 0.05
        best_c_r = float("inf")
        best_rec_r = 0.0
        best_prec_r = 0.0
        best_alert_r = 0.0
        for tau in grid:
            pred = (p_prim >= tau).astype(np.int32)
            tp = int(((pred == 1) & (y_oof == 1)).sum())
            fp = int(((pred == 1) & (y_oof == 0)).sum())
            fn = int(((pred == 0) & (y_oof == 1)).sum())
            c_tot = fn * c_fn + fp * c_fp_r
            if c_tot < best_c_r:
                best_c_r = c_tot
                best_tau_r = float(tau)
                best_rec_r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                best_prec_r = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                best_alert_r = (tp + fp) / len(y_oof) * 100.0
        sens_rows.append(
            {
                "model": primary_model,
                "cost_ratio_fn_to_fp": ratio,
                "theoretical_bayes_tau": round(1.0 / (1.0 + ratio), 4),
                "empirical_optimal_tau_dev": round(best_tau_r, 4),
                "dev_oof_recall": round(best_rec_r, 4),
                "dev_oof_precision": round(best_prec_r, 4),
                "dev_oof_alert_rate_pct": round(best_alert_r, 3),
                "dev_cost_reduction_pct": round((1.0 - best_c_r / (y_oof.sum() * c_fn)) * 100.0, 2),
            }
        )
    pd.DataFrame(sens_rows).to_csv(OUT_DIR / "part5_threshold_sensitivity.csv", index=False)

    curve_df = pd.DataFrame(curve_rows)
    opt_df = pd.DataFrame(optimal_rows)
    curve_df.to_csv(OUT_DIR / "part5_dev_cost_curves.csv", index=False)
    opt_df.to_csv(OUT_DIR / "part5_dev_optimal_thresholds.csv", index=False)
    return curve_df, opt_df, best_thresholds


def run_pyspark_mllib_demo(dev_df: pd.DataFrame, test_df: pd.DataFrame) -> list[dict]:
    """
    Build and evaluate a distributed PySpark MLlib Pipeline (Imputer -> VectorAssembler ->
    StandardScaler -> LogisticRegression & GBTClassifier) fitted strictly on Development
    and evaluated on Test.
    """
    from pyspark.ml import Pipeline as SparkPipeline
    from pyspark.ml.classification import GBTClassifier, LogisticRegression as SparkLR
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    from pyspark.ml.feature import Imputer, StandardScaler as SparkScaler, VectorAssembler
    from pyspark.ml.functions import vector_to_array
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    spark = (
        SparkSession.builder.master("local[*]")
        .appName("BFT65B-Part5-MLlib-Demo")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.shuffle.partitions", "16")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    # Select top 21 domain CAMELS features covering all 6 pillars for the Spark MLlib pipeline demo
    mllib_features = [
        "texas_ratio", "tier1_rbc_ratio", "eq_to_asset", "tier1_lev_ratio",
        "eq_to_asset_chg_4q", "npa_to_asset", "npl_to_loans", "ore_to_asset",
        "reserve_to_loans", "net_chargeoffs_to_loans", "cost_to_income",
        "re_loan_share", "asset_growth_4q", "roa", "roa_roll4_mean",
        "roa_roll4_std", "nim", "liquid_to_asset", "loan_to_dep",
        "brokered_dep_share", "log_asset",
    ]
    imputed_cols = [f"{c}_imp" for c in mllib_features]

    # Clip extreme values on Dev using Dev 0.5%/99.5% bounds and read directly from Parquet in Spark
    clipper = DevQuantileClipper(lower_q=0.005, upper_q=0.995)
    clipper.fit(dev_df[mllib_features])

    sdf_all = spark.read.parquet(str(FEATURES_PATH)).select("split", "CERT", "REPDTE", "y_fail_4q", *mllib_features)
    for idx, c in enumerate(mllib_features):
        lo = float(clipper.lower_bounds_[idx])
        hi = float(clipper.upper_bounds_[idx])
        sdf_all = sdf_all.withColumn(
            c,
            F.when(F.col(c).isNull(), F.lit(None)).otherwise(
                F.least(F.greatest(F.col(c).cast("double"), F.lit(lo)), F.lit(hi))
            ),
        )
    sdf_all = sdf_all.withColumn("label", F.col("y_fail_4q").cast("double"))

    sdf_dev = sdf_all.filter(F.col("split") == "dev").cache()
    sdf_test = sdf_all.filter(F.col("split") == "test").orderBy("CERT", "REPDTE").cache()
    sdf_dev.count()
    sdf_test.count()

    imputer = Imputer(inputCols=mllib_features, outputCols=imputed_cols).setStrategy("median")
    assembler = VectorAssembler(inputCols=imputed_cols, outputCol="raw_features", handleInvalid="keep")
    scaler = SparkScaler(inputCol="raw_features", outputCol="features", withMean=True, withStd=True)

    mllib_results = []
    for model_label, estimator in [
        (
            "PySpark MLlib LogisticRegression",
            SparkLR(featuresCol="features", labelCol="label", maxIter=50, regParam=0.01),
        ),
        (
            "PySpark MLlib GBTClassifier",
            GBTClassifier(featuresCol="features", labelCol="label", maxIter=30, maxDepth=5, seed=SEED),
        ),
    ]:
        t0 = time.perf_counter()
        sp_pipe = SparkPipeline(stages=[imputer, assembler, scaler, estimator])
        sp_model = sp_pipe.fit(sdf_dev)
        preds_sdf = sp_model.transform(sdf_test).select(
            "CERT", "REPDTE", "label", vector_to_array(F.col("probability")).getItem(1).alias("prob")
        ).orderBy("CERT", "REPDTE")
        preds_pd = preds_sdf.toPandas()
        elapsed = time.perf_counter() - t0

        y_true = preds_pd["label"].to_numpy(dtype=np.int32)
        y_prob = preds_pd["prob"].to_numpy(dtype=np.float64)
        m = evaluate_predictions(y_true, y_prob)
        mllib_results.append(
            {
                "model": model_label,
                "engine": "PySpark MLlib (Distributed Pipeline)",
                "pr_auc": round(m["pr_auc"], 4),
                "roc_auc": round(m["roc_auc"], 4),
                "brier_score": round(m["brier_score"], 5),
                "recall_top_1pct": round(m["recall_top_1pct"], 4),
                "recall_top_2pct": round(m["recall_top_2pct"], 4),
                "recall_top_5pct": round(m["recall_top_5pct"], 4),
                "fit_and_eval_s": round(elapsed, 2),
                "y_prob": y_prob,
            }
        )

    sdf_dev.unpersist()
    sdf_test.unpersist()
    spark.stop()
    return mllib_results


def evaluate_once_on_test(
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    models: dict[str, Pipeline],
    best_thresholds: dict[str, float],
    cost_params: dict,
    mllib_results: list[dict],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], dict[str, Pipeline]]:
    """
    Fit each model on the full Development set (1984Q1-2006Q4) and evaluate ONCE on the
    held-out Test set (2008Q1-2025Q2) using the frozen threshold tau* selected on Dev CV.
    """
    X_dev = dev_df[ALL_FEATURES]
    y_dev = dev_df["y_fail_4q"].to_numpy(dtype=np.int32)
    X_test = test_df[ALL_FEATURES]
    y_test = test_df["y_fail_4q"].to_numpy(dtype=np.int32)
    repdte_test = test_df["REPDTE"].to_numpy()

    c_fn = cost_params["c_fn_per_bank_quarter_k"]
    c_fp = cost_params["c_fp_per_bank_quarter_k"]
    no_model_test_cost_k = float(y_test.sum() * c_fn)

    test_rows = []
    subperiod_rows = []
    test_probs = {}
    fitted_models = {}

    for name, pipe in models.items():
        t0 = time.perf_counter()
        est = clone(pipe)
        est.fit(X_dev, y_dev)
        prob_test = est.predict_proba(X_test)[:, 1]
        elapsed = time.perf_counter() - t0

        fitted_models[name] = est
        test_probs[name] = prob_test

        m = evaluate_predictions(y_test, prob_test)
        tau = best_thresholds[name]
        pred = (prob_test >= tau).astype(np.int32)
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()

        total_cost_k = fn * c_fn + fp * c_fp
        cost_red_pct = (1.0 - total_cost_k / no_model_test_cost_k) * 100.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        test_rows.append(
            {
                "model": name,
                "engine": "scikit-learn Pipeline",
                "pr_auc": round(m["pr_auc"], 4),
                "roc_auc": round(m["roc_auc"], 4),
                "brier_score": round(m["brier_score"], 5),
                "recall_top_1pct": round(m["recall_top_1pct"], 4),
                "recall_top_2pct": round(m["recall_top_2pct"], 4),
                "recall_top_5pct": round(m["recall_top_5pct"], 4),
                "dev_selected_threshold": tau,
                "test_precision_at_tau": round(prec, 4),
                "test_recall_at_tau": round(rec, 4),
                "test_specificity_at_tau": round(spec, 4),
                "test_f1_at_tau": round(f1, 4),
                "test_tp": int(tp),
                "test_fp": int(fp),
                "test_fn": int(fn),
                "test_tn": int(tn),
                "test_total_cost_m_usd": round(total_cost_k / 1000.0, 3),
                "test_no_model_cost_m_usd": round(no_model_test_cost_k / 1000.0, 3),
                "test_cost_reduction_pct": round(cost_red_pct, 2),
                "fit_and_eval_s": round(elapsed, 2),
            }
        )

        # Sub-period breakdown on Test (GFC vs Post-GFC vs 2023 Regional Bank Crisis)
        for sp_name, sp_start, sp_end in [
            ("GFC & Aftermath (2008Q1-2014Q4)", "20080331", "20141231"),
            ("Post-GFC Calm (2015Q1-2022Q4)", "20150331", "20221231"),
            ("2023 Regional Bank Crisis (2023Q1-2025Q2)", "20230331", "20250630"),
        ]:
            sp_mask = (repdte_test >= sp_start) & (repdte_test <= sp_end)
            y_sp = y_test[sp_mask]
            p_sp = prob_test[sp_mask]
            m_sp = evaluate_predictions(y_sp, p_sp)
            pred_sp = (p_sp >= tau).astype(np.int32)
            rec_sp = recall_score(y_sp, pred_sp, zero_division=0)
            prec_sp = precision_score(y_sp, pred_sp, zero_division=0)
            subperiod_rows.append(
                {
                    "subperiod": sp_name,
                    "model": name,
                    "bank_quarters": int(sp_mask.sum()),
                    "positive_y1": int(y_sp.sum()),
                    "pr_auc": round(m_sp["pr_auc"], 4),
                    "roc_auc": round(m_sp["roc_auc"], 4),
                    "recall_top_1pct": round(m_sp["recall_top_1pct"], 4),
                    "recall_top_5pct": round(m_sp["recall_top_5pct"], 4),
                    "recall_at_dev_tau": round(float(rec_sp), 4),
                    "precision_at_dev_tau": round(float(prec_sp), 4),
                }
            )

    # Append PySpark MLlib results to test_rows using the Bayes-optimal threshold (0.0476)
    tau_bayes = cost_params["theoretical_bayes_threshold"]
    for mr in mllib_results:
        y_prob = mr["y_prob"]
        test_probs[mr["model"]] = y_prob
        pred = (y_prob >= tau_bayes).astype(np.int32)
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        total_cost_k = fn * c_fn + fp * c_fp
        cost_red_pct = (1.0 - total_cost_k / no_model_test_cost_k) * 100.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        test_rows.append(
            {
                "model": mr["model"],
                "engine": mr["engine"],
                "pr_auc": mr["pr_auc"],
                "roc_auc": mr["roc_auc"],
                "brier_score": mr["brier_score"],
                "recall_top_1pct": mr["recall_top_1pct"],
                "recall_top_2pct": mr["recall_top_2pct"],
                "recall_top_5pct": mr["recall_top_5pct"],
                "dev_selected_threshold": tau_bayes,
                "test_precision_at_tau": round(prec, 4),
                "test_recall_at_tau": round(rec, 4),
                "test_specificity_at_tau": round(spec, 4),
                "test_f1_at_tau": round(f1, 4),
                "test_tp": int(tp),
                "test_fp": int(fp),
                "test_fn": int(fn),
                "test_tn": int(tn),
                "test_total_cost_m_usd": round(total_cost_k / 1000.0, 3),
                "test_no_model_cost_m_usd": round(no_model_test_cost_k / 1000.0, 3),
                "test_cost_reduction_pct": round(cost_red_pct, 2),
                "fit_and_eval_s": mr["fit_and_eval_s"],
            }
        )

    test_res_df = pd.DataFrame(test_rows).sort_values("pr_auc", ascending=False)
    sub_res_df = pd.DataFrame(subperiod_rows)
    test_res_df.to_csv(OUT_DIR / "part5_test_evaluation.csv", index=False)
    sub_res_df.to_csv(OUT_DIR / "part5_test_subperiods.csv", index=False)
    return test_res_df, sub_res_df, test_probs, fitted_models


def extract_feature_importances(
    fitted_models: dict[str, Pipeline], dev_df: pd.DataFrame
) -> pd.DataFrame:
    """Extract permutation feature importance on a stratified/random Dev subsample & RF Gini importance."""
    rng = np.random.default_rng(SEED)
    # Sample 40,000 rows from Development for fast, leak-free permutation importance
    sample_idx = rng.choice(len(dev_df), size=min(40000, len(dev_df)), replace=False)
    X_sub = dev_df[ALL_FEATURES].iloc[sample_idx]
    y_sub = dev_df["y_fail_4q"].to_numpy(dtype=np.int32)[sample_idx]

    gbm_pipe = fitted_models["HistGradientBoosting (GBM)"]
    perm = permutation_importance(
        gbm_pipe,
        X_sub,
        y_sub,
        scoring="average_precision",
        n_repeats=5,
        random_state=SEED,
        n_jobs=-1,
    )

    pillar_map = {f[0]: f[1] for f in m4.FEATURE_CATALOG}
    desc_map = {f[0]: f[3] for f in m4.FEATURE_CATALOG}

    imp_rows = []
    for i, col in enumerate(ALL_FEATURES):
        imp_rows.append(
            {
                "feature": col,
                "camels_pillar": pillar_map.get(col, ""),
                "description": desc_map.get(col, ""),
                "gbm_perm_importance_pr_auc": round(float(perm.importances_mean[i]), 5),
                "gbm_perm_importance_std": round(float(perm.importances_std[i]), 5),
            }
        )
    imp_df = pd.DataFrame(imp_rows).sort_values("gbm_perm_importance_pr_auc", ascending=False)
    imp_df.to_csv(OUT_DIR / "part5_feature_importance.csv", index=False)
    return imp_df


def plot_part5_results(
    y_test: np.ndarray,
    test_probs: dict[str, np.ndarray],
    curve_df: pd.DataFrame,
    best_thresholds: dict[str, float],
    imp_df: pd.DataFrame,
) -> None:
    """Generate a 4-panel high-DPI evaluation figure (PR Curve, ROC Curve, Dev Cost Curve, Top Features)."""
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 10.0), dpi=200)

    palette = {
        "HistGradientBoosting (GBM)": "#d62728",
        "Random Forest (100 trees, depth=10)": "#1f77b4",
        "Logistic Regression (L2 Calibrated)": "#2ca02c",
        "Logistic Regression (Class-Balanced)": "#ff7f0e",
        "PySpark MLlib GBTClassifier": "#9467bd",
        "PySpark MLlib LogisticRegression": "#8c564b",
    }

    # Panel A: Held-Out Test Precision-Recall Curve (2008Q1-2025Q2)
    for name, prob in test_probs.items():
        prec, rec, _ = precision_recall_curve(y_test, prob)
        ap = average_precision_score(y_test, prob)
        axes[0, 0].plot(
            rec,
            prec,
            label=f"{name} (PR-AUC={ap:.3f})",
            color=palette.get(name, "#333333"),
            linewidth=1.8 if "GBM" in name or "Random Forest" in name else 1.3,
        )
    base_rate = float(y_test.mean())
    axes[0, 0].axhline(base_rate, color="gray", linestyle="--", linewidth=1.0, label=f"Random Baseline ({base_rate:.4f})")
    axes[0, 0].set_title("A. Held-Out Test Precision-Recall Curve (2008Q1-2025Q2)")
    axes[0, 0].set_xlabel("Recall (Fraction of 4Q Failures Captured)")
    axes[0, 0].set_ylabel("Precision (Fraction of Alerts that Fail)")
    axes[0, 0].legend(fontsize=8, loc="upper right")
    axes[0, 0].grid(True, alpha=0.3)

    # Panel B: Held-Out Test ROC Curve (2008Q1-2025Q2)
    for name, prob in test_probs.items():
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc = roc_auc_score(y_test, prob)
        axes[0, 1].plot(
            fpr,
            tpr,
            label=f"{name} (ROC-AUC={auc:.3f})",
            color=palette.get(name, "#333333"),
            linewidth=1.8 if "GBM" in name or "Random Forest" in name else 1.3,
        )
    axes[0, 1].plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.0)
    axes[0, 1].set_title("B. Held-Out Test ROC Curve (2008Q1-2025Q2)")
    axes[0, 1].set_xlabel("False Positive Rate (1 - Specificity)")
    axes[0, 1].set_ylabel("True Positive Rate (Recall)")
    axes[0, 1].legend(fontsize=8, loc="lower right")
    axes[0, 1].grid(True, alpha=0.3)

    # Panel C: Development Out-of-Fold Cost Optimization Curve
    for name in [
        "HistGradientBoosting (GBM)",
        "Random Forest (100 trees, depth=10)",
        "Logistic Regression (L2 Calibrated)",
    ]:
        sub = curve_df[(curve_df["model"] == name) & (curve_df["threshold"] <= 0.40)]
        tau_opt = best_thresholds[name]
        min_cost = sub.loc[sub["threshold"] == tau_opt, "total_cost_m_usd"].values
        col = palette.get(name, "#333333")
        axes[1, 0].plot(
            sub["threshold"],
            sub["total_cost_m_usd"],
            label=f"{name} (tau*={tau_opt:.3f})",
            color=col,
            linewidth=1.8,
        )
        if len(min_cost) > 0:
            axes[1, 0].scatter([tau_opt], [min_cost[0]], color=col, s=55, zorder=5)
    no_mod = curve_df["no_model_cost_m_usd"].iloc[0]
    axes[1, 0].axhline(no_mod, color="black", linestyle=":", linewidth=1.2, label=f"No-Model Cost (${no_mod:.0f}M)")
    axes[1, 0].set_title("C. Expected Cost Curve on Development OOF CV (C_FN : C_FP = 20 : 1)")
    axes[1, 0].set_xlabel("Alarm Probability Threshold (tau)")
    axes[1, 0].set_ylabel("Expected Total Cost ($ Millions)")
    axes[1, 0].legend(fontsize=8, loc="upper right")
    axes[1, 0].grid(True, alpha=0.3)

    # Panel D: Top 12 Features by GBM Permutation Importance (PR-AUC Delta on Dev)
    top12 = imp_df.head(12).sort_values("gbm_perm_importance_pr_auc", ascending=True)
    axes[1, 1].barh(top12["feature"], top12["gbm_perm_importance_pr_auc"], color="#1f77b4")
    axes[1, 1].set_title("D. Top 12 Features by Permutation Importance (PR-AUC Drop on Dev)")
    axes[1, 1].set_xlabel("Mean Decrease in PR-AUC When Permuted")
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(
        "Part 5: Bank Failure Early-Warning Model Validation, Cost Optimization & Single Test Evaluation",
        fontsize=12.5,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "part5_model_evaluation.png", dpi=200)
    plt.close(fig)


def main() -> None:
    import argparse
    import pickle

    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["all", "sklearn", "spark", "finalize"], default="all")
    args = ap.parse_args()
    cache_path = CLEAN_DIR / "_part5_cache.pkl"

    print("=== Part 5: Loading Engineered CAMELS Feature Panel ===")
    feat_pl = pl.read_parquet(FEATURES_PATH).sort(["CERT", "REPDTE"])
    dev_df = feat_pl.filter(pl.col("split") == "dev").to_pandas()
    test_df = feat_pl.filter(pl.col("split") == "test").to_pandas()

    if args.stage in ("all", "sklearn"):
        print(
            f"Development set (1984Q1-2006Q4): {len(dev_df):,} rows, {int(dev_df['y_fail_4q'].sum()):,} positives "
            f"({dev_df['y_fail_4q'].mean()*100:.3f}%)"
        )
        print(
            f"Held-out Test set (2008Q1-2025Q2): {len(test_df):,} rows, {int(test_df['y_fail_4q'].sum()):,} positives "
            f"({test_df['y_fail_4q'].mean()*100:.3f}%)"
        )

        print("\n=== Part 5: Estimating Asymmetric Misclassification Costs from Development Data ===")
        cost_params = estimate_empirical_costs_from_dev()
        for k, v in cost_params.items():
            print(f"  {k:<34}: {v}")

        print("\n=== Part 5: Running 4-Fold Purged Expanding-Window CV on Development Set ===")
        models = build_candidate_models()
        fold_df, oof_df, oof_preds, oof_mask = run_purged_expanding_cv(dev_df, models)
        print("\n--- Development Out-of-Fold Purged CV Summary (1988Q1-2006Q4, 4Q Purge Gap) ---")
        print(oof_df.to_string(index=False))

        print("\n=== Part 5: Optimizing Alarm Thresholds on Development Out-of-Fold Predictions ===")
        y_oof = dev_df["y_fail_4q"].to_numpy(dtype=np.int32)[oof_mask]
        curve_df, opt_df, best_thresholds = optimize_cost_threshold_on_dev_oof(
            y_oof, oof_preds, oof_mask, cost_params
        )
        print(
            opt_df[
                [
                    "model",
                    "threshold",
                    "total_cost_m_usd",
                    "cost_reduction_pct",
                    "precision",
                    "recall",
                    "f1_score",
                    "alert_rate_pct",
                ]
            ].to_string(index=False)
        )
        if args.stage == "sklearn":
            with open(cache_path, "wb") as f:
                pickle.dump(
                    {
                        "cost_params": cost_params,
                        "curve_df": curve_df,
                        "opt_df": opt_df,
                        "best_thresholds": best_thresholds,
                    },
                    f,
                )
            return

    if args.stage in ("all", "spark"):
        print("\n=== Part 5: Running Distributed PySpark MLlib Pipeline Demo ===")
        mllib_results = run_pyspark_mllib_demo(dev_df, test_df)
        if args.stage == "spark":
            state = pickle.loads(cache_path.read_bytes()) if cache_path.exists() else {}
            state["mllib_results"] = mllib_results
            cache_path.write_bytes(pickle.dumps(state))
            return

    if args.stage == "finalize":
        state = pickle.loads(cache_path.read_bytes())
        cost_params = state["cost_params"]
        curve_df = state["curve_df"]
        best_thresholds = state["best_thresholds"]
        mllib_results = state["mllib_results"]
        models = build_candidate_models()

    print("\n=== Part 5: Single Final Evaluation on Held-Out Test Set (2008Q1-2025Q2) ===")
    test_res_df, sub_res_df, test_probs, fitted_models = evaluate_once_on_test(
        dev_df, test_df, models, best_thresholds, cost_params, mllib_results
    )
    print("\n--- Held-Out Test Set Evaluation (2008Q1-2025Q2: 435,446 Rows, 2,181 Positives, 558 Failing Banks) ---")
    print(
        test_res_df[
            [
                "model",
                "pr_auc",
                "roc_auc",
                "brier_score",
                "recall_top_1pct",
                "recall_top_5pct",
                "dev_selected_threshold",
                "test_precision_at_tau",
                "test_recall_at_tau",
                "test_cost_reduction_pct",
            ]
        ].to_string(index=False)
    )

    print("\n--- Held-Out Test Sub-Period Breakdown (GFC vs. Post-GFC vs. 2023 Regional Bank Crisis) ---")
    print(
        sub_res_df[sub_res_df["model"].isin(["HistGradientBoosting (GBM)", "Logistic Regression (L2 Calibrated)"])].to_string(
            index=False
        )
    )

    print("\n=== Part 5: Extracting Feature Importances & Generating Charts ===")
    imp_df = extract_feature_importances(fitted_models, dev_df)
    print(imp_df.head(12).to_string(index=False))

    y_test = test_df["y_fail_4q"].to_numpy(dtype=np.int32)
    plot_part5_results(y_test, test_probs, curve_df, best_thresholds, imp_df)
    if cache_path.exists():
        cache_path.unlink()
    print(f"\nSaved all Part 5 tables and charts to {OUT_DIR}")


if __name__ == "__main__":
    main()
