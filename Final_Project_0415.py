"""
GLORE Federated Logistic + Baselines + Runtime + Blockchain Payloads + Visualizations
Defaults:
  BASE_DIR = "/Users/xiyuehuang/Desktop/CBB 5790"
  DATA_PATH = BASE_DIR + "/multi_PRS_dataset.csv"
  OUT_DIR  = BASE_DIR + "/outputs_glore_blockchain"
No required CLI args.
"""

import argparse
import json
import math
import os
import time
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier


# -----------------------------
# Defaults (YOU asked for these)
# -----------------------------
BASE_DIR_DEFAULT = "/Users/xiyuehuang/Desktop/CBB 5790"
DATA_PATH_DEFAULT = os.path.join(BASE_DIR_DEFAULT, "multi_PRS_dataset.csv")
OUT_DIR_DEFAULT = os.path.join(BASE_DIR_DEFAULT, "outputs_glore_blockchain")


# -----------------------------
# Metrics
# -----------------------------
def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    probs = np.asarray(probs).reshape(-1)
    y_true = np.asarray(y_true).reshape(-1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0
    n = len(probs)
    for b0, b1 in zip(bins[:-1], bins[1:]):
        mask = (probs >= b0) & (probs < b1) if b1 < 1.0 else (probs >= b0) & (probs <= b1)
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        conf = float(probs[mask].mean())
        acc = float(y_true[mask].mean())
        ece += (cnt / n) * abs(acc - conf)
    return float(ece)


def reliability_bins(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10):
    probs = np.asarray(probs).reshape(-1)
    y_true = np.asarray(y_true).reshape(-1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)

    bin_conf, bin_acc, bin_cnt = [], [], []
    for b0, b1 in zip(bins[:-1], bins[1:]):
        mask = (probs >= b0) & (probs < b1) if b1 < 1.0 else (probs >= b0) & (probs <= b1)
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        bin_conf.append(float(probs[mask].mean()))
        bin_acc.append(float(y_true[mask].mean()))
        bin_cnt.append(cnt)
    return np.array(bin_conf), np.array(bin_acc), np.array(bin_cnt)


def safe_auroc(y_true: np.ndarray, probs: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    probs = np.asarray(probs).reshape(-1)
    try:
        return float(roc_auc_score(y_true, probs))
    except ValueError:
        return float("nan")


def float_to_x1e6(x: float) -> int:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return 0
    return int(round(float(x) * 1_000_000))


def sha256_hex_to_bytes32(hex_str: str) -> str:
    if hex_str.startswith("0x"):
        hex_str = hex_str[2:]
    if len(hex_str) < 64:
        hex_str = hex_str.ljust(64, "0")
    return "0x" + hex_str[:64]


# -----------------------------
# Data splitting
# -----------------------------
def set_seed(seed: int):
    np.random.seed(seed)


def make_random_sites(df: pd.DataFrame, site_names: List[str], seed: int) -> Dict[str, pd.DataFrame]:
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    splits = np.array_split(df, len(site_names))
    return {site_names[i]: splits[i].reset_index(drop=True) for i in range(len(site_names))}


def train_test_split_df(df: pd.DataFrame, train_frac: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df)
    n_train = int(train_frac * n)
    return df.iloc[:n_train].copy(), df.iloc[n_train:].copy()


@dataclass
class SiteData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def df_to_xy(df: pd.DataFrame, feature_cols: List[str], label_col: str) -> Tuple[np.ndarray, np.ndarray]:
    X = df[feature_cols].astype(np.float64).values
    y = df[label_col].astype(np.float64).values
    return X, y


# -----------------------------
# Centralized baseline models
# -----------------------------
def fit_eval_model(name: str, model, Xtr, ytr, Xte, yte) -> Dict[str, float]:
    t0 = time.perf_counter()
    model.fit(Xtr, ytr)
    train_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    probs = model.predict_proba(Xte)[:, 1]
    infer_time = time.perf_counter() - t1

    return {
        "model": name,
        "auroc": safe_auroc(yte, probs),
        "ece_10bin": expected_calibration_error(probs, yte, n_bins=10),
        "train_time_s": float(train_time),
        "infer_time_s": float(infer_time),
        "total_time_s": float(train_time + infer_time),
    }


# -----------------------------
# GLORE (Federated Logistic Regression)
# -----------------------------
def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))


def add_intercept(X: np.ndarray) -> np.ndarray:
    return np.hstack([np.ones((X.shape[0], 1), dtype=X.dtype), X])


def logistic_grad_hess(beta: np.ndarray, X: np.ndarray, y: np.ndarray, l2: float) -> Tuple[np.ndarray, np.ndarray]:
    z = X @ beta
    p = sigmoid(z)
    g = X.T @ (p - y)
    w = p * (1.0 - p)
    H = X.T @ (X * w[:, None])

    g[1:] += l2 * beta[1:]
    H[1:, 1:] += l2 * np.eye(H.shape[0] - 1)
    return g, H


def beta_hash_sha256(beta: np.ndarray) -> str:
    return hashlib.sha256(beta.astype(np.float64).tobytes()).hexdigest()


def predict_beta(beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    return sigmoid(add_intercept(X) @ beta)


def evaluate_beta(beta: np.ndarray, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    probs = predict_beta(beta, X)
    return {
        "auroc": safe_auroc(y, probs),
        "ece_10bin": expected_calibration_error(probs, y, n_bins=10),
        "pos_rate": float(np.mean(y)),
        "avg_prob": float(np.mean(probs)),
    }


def glore_train(sites: Dict[str, SiteData], rounds: int, l2: float, damping: float, out_dir: str):
    any_site = next(iter(sites.values()))
    d = any_site.X_train.shape[1]
    beta = np.zeros(d + 1, dtype=np.float64)

    payload_path = os.path.join(out_dir, "chain_payloads.jsonl")
    with open(payload_path, "w") as _:
        pass

    round_logs = []
    for r in range(1, rounds + 1):
        t0 = time.perf_counter()

        g_sum = np.zeros_like(beta)
        H_sum = np.zeros((beta.size, beta.size), dtype=np.float64)

        for sd in sites.values():
            X_i = add_intercept(sd.X_train)
            g, H = logistic_grad_hess(beta, X_i, sd.y_train, l2=l2)
            g_sum += g
            H_sum += H

        H_damped = H_sum + damping * np.eye(H_sum.shape[0])
        step = np.linalg.solve(H_damped, g_sum)
        beta = beta - step

        per_site = {s: evaluate_beta(beta, sd.X_test, sd.y_test) for s, sd in sites.items()}
        aucs = [per_site[s]["auroc"] for s in per_site if not math.isnan(per_site[s]["auroc"])]
        eces = [per_site[s]["ece_10bin"] for s in per_site if not math.isnan(per_site[s]["ece_10bin"])]
        avg_auc = float(np.mean(aucs)) if len(aucs) else float("nan")
        avg_ece = float(np.mean(eces)) if len(eces) else float("nan")

        ghash = beta_hash_sha256(beta)
        payload = {
            "round": r,
            "global_model_hash_sha256": ghash,
            "bytes32_modelHash": sha256_hex_to_bytes32(ghash),
            "avg_auc": avg_auc,
            "avg_ece": avg_ece,
            "avgAuc_x1e6": float_to_x1e6(avg_auc),
            "avgEce_x1e6": float_to_x1e6(avg_ece),
            "per_site_test": per_site,
            "round_runtime_s": float(time.perf_counter() - t0),
        }
        with open(payload_path, "a") as f:
            f.write(json.dumps(payload) + "\n")
        round_logs.append(payload)

    return beta, round_logs


# -----------------------------
# Plot helpers
# -----------------------------
def savefig(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_label_prevalence(y: np.ndarray, out_path: str):
    pos = float(np.mean(y))
    plt.figure()
    plt.bar([0, 1], [1 - pos, pos])
    plt.xticks([0, 1], ["label=0", "label=1"])
    plt.ylabel("Proportion")
    plt.title("Label prevalence")
    savefig(out_path)


def plot_feature_distributions(df: pd.DataFrame, feature_cols: List[str], out_path: str, max_features: int = 6):
    cols = feature_cols[:max_features]
    plt.figure(figsize=(8, 6))
    for c in cols:
        plt.hist(df[c].dropna().values, bins=30, alpha=0.5, label=c)
    plt.legend(fontsize=8)
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.title(f"Feature distributions (showing {len(cols)} of {len(feature_cols)})")
    savefig(out_path)


def plot_corr_heatmap(df: pd.DataFrame, feature_cols: List[str], out_path: str, max_features: int = 20):
    cols = feature_cols[:max_features]
    corr = df[cols].corr().values
    plt.figure(figsize=(7, 6))
    plt.imshow(corr, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(cols)), cols, rotation=90, fontsize=6)
    plt.yticks(range(len(cols)), cols, fontsize=6)
    plt.title(f"PRS correlation heatmap (top {len(cols)})")
    savefig(out_path)


def plot_model_auroc_runtime(df: pd.DataFrame, out_path: str):
    d = df.sort_values("auroc", ascending=False).copy()
    x = np.arange(len(d))
    plt.figure(figsize=(8, 4))
    plt.bar(x - 0.2, d["auroc"].values, width=0.4, label="AUROC")
    plt.bar(x + 0.2, d["total_time_s"].values, width=0.4, label="Runtime (s)")
    plt.xticks(x, d["model"].values, rotation=20, ha="right")
    plt.title("Baseline models: AUROC vs runtime")
    plt.legend()
    savefig(out_path)


def plot_model_runtime_breakdown(df: pd.DataFrame, out_path: str):
    d = df.sort_values("total_time_s", ascending=False).copy()
    x = np.arange(len(d))
    plt.figure(figsize=(8, 4))
    plt.bar(x, d["train_time_s"].values, label="Train time (s)")
    plt.bar(x, d["infer_time_s"].values, bottom=d["train_time_s"].values, label="Inference time (s)")
    plt.xticks(x, d["model"].values, rotation=20, ha="right")
    plt.title("Baseline runtime breakdown")
    plt.legend()
    savefig(out_path)


def plot_glore_roc(beta: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, out_path: str):
    probs = predict_beta(beta, X_test)
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = safe_auroc(y_test, probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("GLORE Logistic: ROC (pooled test)")
    plt.legend()
    savefig(out_path)


def plot_glore_calibration(beta: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, out_path: str):
    probs = predict_beta(beta, X_test)
    conf, acc, _ = reliability_bins(probs, y_test, n_bins=10)
    plt.figure()
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.plot(conf, acc, marker="o", label="GLORE")
    plt.xlabel("Mean predicted probability (bin)")
    plt.ylabel("Empirical positive rate (bin)")
    plt.title("GLORE Logistic: Reliability diagram (pooled test)")
    plt.legend()
    savefig(out_path)


def plot_glore_per_site(per_site: Dict[str, Dict[str, float]], out_path: str):
    sites = list(per_site.keys())
    aucs = [per_site[s]["auroc"] for s in sites]
    eces = [per_site[s]["ece_10bin"] for s in sites]
    x = np.arange(len(sites))
    plt.figure(figsize=(7, 4))
    plt.bar(x - 0.2, aucs, width=0.4, label="AUROC")
    plt.bar(x + 0.2, eces, width=0.4, label="ECE (10-bin)")
    plt.xticks(x, sites)
    plt.title("GLORE performance by site (test)")
    plt.legend()
    savefig(out_path)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, default=DATA_PATH_DEFAULT)
    ap.add_argument("--out_dir", type=str, default=OUT_DIR_DEFAULT)
    ap.add_argument("--label_col", type=str, default="label")
    ap.add_argument("--iid_col", type=str, default="IID")
    ap.add_argument("--site_names", type=str, default="SiteA,SiteB,SiteC")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--glore_rounds", type=int, default=30)
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--damping", type=float, default=1e-6)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    fig_dir = os.path.join(args.out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    set_seed(args.seed)
    df = pd.read_csv(args.data_path)

    # features
    feature_cols = [c for c in df.columns if c.endswith("_PRS_z")]
    if len(feature_cols) == 0:
        feature_cols = [
            c for c in df.columns
            if c not in [args.iid_col, args.label_col] and pd.api.types.is_numeric_dtype(df[c])
        ]
    if args.label_col not in df.columns:
        raise ValueError(f"label_col='{args.label_col}' not found. Columns={list(df.columns)}")

    # data description figs
    y_all = df[args.label_col].astype(float).values
    plot_label_prevalence(y_all, os.path.join(fig_dir, "label_prevalence.png"))
    plot_feature_distributions(df, feature_cols, os.path.join(fig_dir, "feature_distributions.png"), max_features=6)
    plot_corr_heatmap(df, feature_cols, os.path.join(fig_dir, "prs_correlation_heatmap.png"), max_features=min(20, len(feature_cols)))

    # sites
    site_names = [s.strip() for s in args.site_names.split(",") if s.strip()]
    sites_raw = make_random_sites(df, site_names, seed=args.seed)

    sites: Dict[str, SiteData] = {}
    for i, s in enumerate(site_names):
        tr_df, te_df = train_test_split_df(sites_raw[s], train_frac=args.train_frac, seed=args.seed + i + 10)
        Xtr, ytr = df_to_xy(tr_df, feature_cols, args.label_col)
        Xte, yte = df_to_xy(te_df, feature_cols, args.label_col)
        sites[s] = SiteData(X_train=Xtr, y_train=ytr, X_test=Xte, y_test=yte)

    X_train_all = np.vstack([sites[s].X_train for s in site_names])
    y_train_all = np.concatenate([sites[s].y_train for s in site_names])
    X_test_all = np.vstack([sites[s].X_test for s in site_names])
    y_test_all = np.concatenate([sites[s].y_test for s in site_names])

    # baselines + runtime
    rows = []
    rows.append(fit_eval_model("LogisticRegression", LogisticRegression(max_iter=2000, solver="lbfgs"),
                               X_train_all, y_train_all, X_test_all, y_test_all))
    rows.append(fit_eval_model("RandomForest", RandomForestClassifier(n_estimators=300, random_state=args.seed, n_jobs=-1),
                               X_train_all, y_train_all, X_test_all, y_test_all))
    rows.append(fit_eval_model("GradientBoosting", GradientBoostingClassifier(random_state=args.seed),
                               X_train_all, y_train_all, X_test_all, y_test_all))
    rows.append(fit_eval_model("MLP", MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=args.seed),
                               X_train_all, y_train_all, X_test_all, y_test_all))

    baseline_df = pd.DataFrame(rows)
    baseline_csv = os.path.join(args.out_dir, "baseline_models_with_runtime.csv")
    baseline_df.to_csv(baseline_csv, index=False)

    plot_model_auroc_runtime(baseline_df, os.path.join(fig_dir, "model_auroc_runtime.png"))
    plot_model_runtime_breakdown(baseline_df, os.path.join(fig_dir, "model_runtime_breakdown.png"))

    # glore
    t0 = time.perf_counter()
    beta, round_logs = glore_train(sites, rounds=args.glore_rounds, l2=args.l2, damping=args.damping, out_dir=args.out_dir)
    glore_time = time.perf_counter() - t0

    final_per_site = {s: evaluate_beta(beta, sites[s].X_test, sites[s].y_test) for s in site_names}
    final_pooled = evaluate_beta(beta, X_test_all, y_test_all)

    plot_glore_roc(beta, X_test_all, y_test_all, os.path.join(fig_dir, "glore_roc_pooled.png"))
    plot_glore_calibration(beta, X_test_all, y_test_all, os.path.join(fig_dir, "glore_calibration_pooled.png"))
    plot_glore_per_site(final_per_site, os.path.join(fig_dir, "glore_per_site_auroc_ece.png"))

    summary = {
        "base_dir": BASE_DIR_DEFAULT,
        "data_path": args.data_path,
        "out_dir": args.out_dir,
        "figures_dir": fig_dir,
        "features_used": feature_cols,
        "sites": site_names,
        "train_frac_per_site": args.train_frac,
        "baseline_models_with_runtime_csv": baseline_csv,
        "glore_total_runtime_s": float(glore_time),
        "glore_final_hash_sha256": beta_hash_sha256(beta),
        "glore_final_bytes32_modelHash": sha256_hex_to_bytes32(beta_hash_sha256(beta)),
        "glore_final_per_site_test": final_per_site,
        "glore_final_pooled_test": final_pooled,
        "blockchain_payloads_jsonl": os.path.join(args.out_dir, "chain_payloads.jsonl")
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nDONE. Outputs saved under:")
    print(" -", args.out_dir)
    print(" - figures:", fig_dir)
    print("\nFinal round payload (paste into geth):")
    print(json.dumps(round_logs[-1], indent=2))


if __name__ == "__main__":
    main()