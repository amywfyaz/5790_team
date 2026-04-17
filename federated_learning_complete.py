"""
Complete Federated Learning Pipeline for CAD Polygenic Risk Prediction
=====================================================================

Implements multiple FL strategies with comprehensive evaluation:
  - FedAvg  (Federated Averaging with Logistic Regression & MLP)
  - FedProx (proximal regularization for non-IID robustness)
  - Differential Privacy (Gaussian mechanism with gradient clipping)
  - GLORE   (exact federated logistic regression, imported if available)

Data partitioning schemes:
  - IID (stratified random)
  - Non-IID label skew (Dirichlet-based)
  - Non-IID feature shift (PRS-quantile-based)

Evaluation metrics: AUROC, Brier Score, ECE, Accuracy, LogLoss

Usage
-----
    python federated_learning_complete.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

# ── Try importing existing GLORE module ──────────────────────────────────
_TEAM_DIR = Path(__file__).resolve().parent / "5790_team-main"
sys.path.insert(0, str(_TEAM_DIR))
try:
    from federated_glore import (
        FederatedLogisticRegression as GLOREModel,
        LocalSite as GLORESite,
        evaluate_predictions as glore_eval,
    )
    HAS_GLORE = True
except ImportError:
    HAS_GLORE = False

# ── Configuration ────────────────────────────────────────────────────────
MULTI_PRS_PATH = _TEAM_DIR / "multi_PRS_dataset.csv"
PRS_PATH = _TEAM_DIR / "PRS_dataset.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "fl_results"

FEATURE_COLS = ["CAD_PRS_z", "BMI_PRS_z", "HTN_PRS_z"]
LABEL_COL = "CAD_status"

N_SITES = 4
SEED = 42
TEST_RATIO = 0.25

FL_ROUNDS = 30
LOCAL_EPOCHS = 5
LR = 0.02
L2 = 1e-4
FEDPROX_MU = 0.1

DP_CLIP_NORM = 1.0
DP_NOISE_MULT = 0.5
DP_DELTA = 1e-5

HIDDEN_SIZES = [32, 16]

# ── Utilities ────────────────────────────────────────────────────────────

def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)


def relu_grad(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(float)


# ── Data splitting ───────────────────────────────────────────────────────

def load_data(path: Path = MULTI_PRS_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def split_train_test(
    df: pd.DataFrame, test_ratio: float = TEST_RATIO, seed: int = SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_test = int(len(df) * test_ratio)
    return (
        df.iloc[idx[n_test:]].reset_index(drop=True),
        df.iloc[idx[:n_test]].reset_index(drop=True),
    )


def iid_split(
    df: pd.DataFrame, n_sites: int = N_SITES, seed: int = SEED,
) -> List[pd.DataFrame]:
    """Stratified IID split across sites."""
    rng = np.random.default_rng(seed)
    buckets: List[list] = [[] for _ in range(n_sites)]
    for _, grp in df.groupby(LABEL_COL):
        idx = rng.permutation(len(grp))
        for i, part in enumerate(np.array_split(grp.iloc[idx], n_sites)):
            buckets[i].append(part)
    return [
        pd.concat(b).sample(frac=1, random_state=seed).reset_index(drop=True)
        for b in buckets
    ]


def non_iid_label_skew(
    df: pd.DataFrame, n_sites: int = N_SITES, alpha: float = 0.5, seed: int = SEED,
) -> List[pd.DataFrame]:
    """Dirichlet-based label-skew split.  Lower alpha ⇒ more heterogeneous."""
    rng = np.random.default_rng(seed)
    site_idx: List[list] = [[] for _ in range(n_sites)]
    for label in df[LABEL_COL].unique():
        rows = df.index[df[LABEL_COL] == label].tolist()
        rng.shuffle(rows)
        props = rng.dirichlet(np.full(n_sites, alpha))
        counts = (props * len(rows)).astype(int)
        counts[-1] = len(rows) - counts[:-1].sum()
        cur = 0
        for i, c in enumerate(counts):
            site_idx[i].extend(rows[cur : cur + c])
            cur += c
    return [df.iloc[si].reset_index(drop=True) for si in site_idx]


def non_iid_feature_shift(
    df: pd.DataFrame, n_sites: int = N_SITES, feature: str = "CAD_PRS_z",
) -> List[pd.DataFrame]:
    """Quantile-based feature shift (sites see different PRS ranges)."""
    return [
        chunk.reset_index(drop=True)
        for chunk in np.array_split(df.sort_values(feature).reset_index(drop=True), n_sites)
    ]


# ── Models (numpy-only) ─────────────────────────────────────────────────

Params = Dict[str, np.ndarray]


class LogReg:
    """Logistic regression implemented in numpy."""

    def __init__(self, d: int) -> None:
        self.W = np.zeros(d)
        self.b = 0.0

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(X @ self.W + self.b)

    def step(self, X: np.ndarray, y: np.ndarray, lr: float = LR,
             l2: float = L2, mu: float = 0.0, ref: Optional[Params] = None) -> None:
        n = len(y)
        err = self.predict_proba(X) - y
        gW = X.T @ err / n + l2 * self.W
        gb = err.mean()
        if mu > 0 and ref is not None:
            gW += mu * (self.W - ref["W"])
            gb += mu * (self.b - ref["b"][0])
        self.W -= lr * gW
        self.b -= lr * gb

    def get_params(self) -> Params:
        return {"W": self.W.copy(), "b": np.array([self.b])}

    def set_params(self, p: Params) -> None:
        self.W = p["W"].copy()
        self.b = float(p["b"][0])


class MLP:
    """Multi-layer perceptron (ReLU hidden, sigmoid output)."""

    def __init__(self, d: int, hidden: List[int] = HIDDEN_SIZES) -> None:
        rng = np.random.default_rng(SEED)
        sizes = [d, *hidden, 1]
        self.layers = []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            self.layers.append({
                "W": rng.normal(0, np.sqrt(2 / fan_in), (fan_in, fan_out)),
                "b": np.zeros(fan_out),
            })

    def _forward(self, X: np.ndarray):
        cache, a = [], X
        for i, L in enumerate(self.layers):
            z = a @ L["W"] + L["b"]
            cache.append((a, z))
            a = sigmoid(z) if i == len(self.layers) - 1 else relu(z)
        return a.ravel(), cache

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._forward(X)[0]

    def step(self, X: np.ndarray, y: np.ndarray, lr: float = LR,
             l2: float = L2, mu: float = 0.0, ref: Optional[List[dict]] = None) -> None:
        n = len(y)
        pred, cache = self._forward(X)
        dz = (pred - y).reshape(-1, 1)

        for i in reversed(range(len(self.layers))):
            a_prev, z = cache[i]
            dW = a_prev.T @ dz / n + l2 * self.layers[i]["W"]
            db = dz.mean(axis=0)
            if mu > 0 and ref is not None:
                dW += mu * (self.layers[i]["W"] - ref[i]["W"])
                db += mu * (self.layers[i]["b"] - ref[i]["b"])
            if i > 0:
                dz = (dz @ self.layers[i]["W"].T) * relu_grad(cache[i - 1][1])
            self.layers[i]["W"] -= lr * dW
            self.layers[i]["b"] -= lr * db

    def get_params(self) -> Params:
        p: Params = {}
        for i, L in enumerate(self.layers):
            p[f"W{i}"] = L["W"].copy()
            p[f"b{i}"] = L["b"].copy()
        return p

    def set_params(self, p: Params) -> None:
        for i in range(len(self.layers)):
            self.layers[i]["W"] = p[f"W{i}"].copy()
            self.layers[i]["b"] = p[f"b{i}"].copy()

    def get_ref(self) -> List[dict]:
        return [{"W": L["W"].copy(), "b": L["b"].copy()} for L in self.layers]


def _make_model(kind: str, d: int):
    return MLP(d) if kind == "mlp" else LogReg(d)


# ── Differential privacy helpers ─────────────────────────────────────────

def clip_update(local_p: Params, global_p: Params, S: float = DP_CLIP_NORM) -> Params:
    delta = {k: local_p[k] - global_p[k] for k in local_p}
    norm = np.linalg.norm(np.concatenate([v.ravel() for v in delta.values()]))
    scale = min(1.0, S / (norm + 1e-12))
    return {k: global_p[k] + delta[k] * scale for k in delta}


def add_noise(params: Params, S: float = DP_CLIP_NORM,
              sigma_mult: float = DP_NOISE_MULT, seed: int = 0) -> Params:
    rng = np.random.default_rng(seed)
    sigma = S * sigma_mult
    return {k: v + rng.normal(0, sigma, v.shape) for k, v in params.items()}


def estimate_epsilon(sigma_mult: float, T: int, delta: float = DP_DELTA) -> float:
    if sigma_mult <= 0:
        return float("inf")
    eps1 = np.sqrt(2 * np.log(1.25 / delta)) / sigma_mult
    return eps1 * np.sqrt(T)


# ── Federated algorithms ─────────────────────────────────────────────────

def weighted_avg(param_list: List[Params], sizes: List[int]) -> Params:
    total = sum(sizes)
    return {
        k: sum(p[k] * (n / total) for p, n in zip(param_list, sizes))
        for k in param_list[0]
    }


SiteData = List[Tuple[np.ndarray, np.ndarray]]


def run_fedavg(
    sites: SiteData, X_te: np.ndarray, y_te: np.ndarray,
    kind: str = "mlp", rounds: int = FL_ROUNDS, E: int = LOCAL_EPOCHS,
    use_dp: bool = False,
) -> Tuple[object, List[dict]]:
    d = sites[0][0].shape[1]
    gm = _make_model(kind, d)
    history = []
    t_start = time.perf_counter()

    for r in range(1, rounds + 1):
        t_round = time.perf_counter()
        gp = gm.get_params()
        locals_, ns = [], []
        for Xs, ys in sites:
            lm = _make_model(kind, d)
            lm.set_params(gp)
            for _ in range(E):
                lm.step(Xs, ys)
            lp = lm.get_params()
            if use_dp:
                lp = clip_update(lp, gp)
            locals_.append(lp)
            ns.append(len(ys))

        agg = weighted_avg(locals_, ns)
        if use_dp:
            agg = add_noise(agg, seed=SEED + r)
        gm.set_params(agg)

        elapsed_round = time.perf_counter() - t_round
        history.append({
            "round": r,
            "round_time_sec": round(elapsed_round, 4),
            **compute_metrics(y_te, gm.predict_proba(X_te)),
        })

    total_time = time.perf_counter() - t_start
    for h in history:
        h["total_time_sec"] = round(total_time, 4)
    return gm, history


def run_fedprox(
    sites: SiteData, X_te: np.ndarray, y_te: np.ndarray,
    kind: str = "mlp", rounds: int = FL_ROUNDS, E: int = LOCAL_EPOCHS,
    mu: float = FEDPROX_MU,
) -> Tuple[object, List[dict]]:
    d = sites[0][0].shape[1]
    gm = _make_model(kind, d)
    history = []
    t_start = time.perf_counter()

    for r in range(1, rounds + 1):
        t_round = time.perf_counter()
        gp = gm.get_params()
        ref = gm.get_ref() if kind == "mlp" else gp
        locals_, ns = [], []
        for Xs, ys in sites:
            lm = _make_model(kind, d)
            lm.set_params(gp)
            for _ in range(E):
                lm.step(Xs, ys, mu=mu, ref=ref)
            locals_.append(lm.get_params())
            ns.append(len(ys))

        gm.set_params(weighted_avg(locals_, ns))
        elapsed_round = time.perf_counter() - t_round
        history.append({
            "round": r,
            "round_time_sec": round(elapsed_round, 4),
            **compute_metrics(y_te, gm.predict_proba(X_te)),
        })

    total_time = time.perf_counter() - t_start
    for h in history:
        h["total_time_sec"] = round(total_time, 4)
    return gm, history


def run_local(
    sites: SiteData, X_te: np.ndarray, y_te: np.ndarray,
    kind: str = "mlp", total_epochs: int = FL_ROUNDS * LOCAL_EPOCHS,
) -> List[dict]:
    d = sites[0][0].shape[1]
    results = []
    t_start = time.perf_counter()
    for i, (Xs, ys) in enumerate(sites):
        t_site = time.perf_counter()
        m = _make_model(kind, d)
        for _ in range(total_epochs):
            m.step(Xs, ys)
        site_time = time.perf_counter() - t_site
        results.append({
            "site": i + 1,
            "site_time_sec": round(site_time, 4),
            **compute_metrics(y_te, m.predict_proba(X_te)),
        })
    total_time = time.perf_counter() - t_start
    for r in results:
        r["total_time_sec"] = round(total_time, 4)
    return results


def run_centralized(
    X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, y_te: np.ndarray,
    kind: str = "mlp", total_epochs: int = FL_ROUNDS * LOCAL_EPOCHS,
) -> dict:
    t_start = time.perf_counter()
    m = _make_model(kind, X_tr.shape[1])
    for _ in range(total_epochs):
        m.step(X_tr, y_tr)
    total_time = time.perf_counter() - t_start
    return {
        "total_time_sec": round(total_time, 4),
        **compute_metrics(y_te, m.predict_proba(X_te)),
    }


def run_glore(
    train_df: pd.DataFrame, test_df: pd.DataFrame,
) -> Optional[dict]:
    """Run GLORE using existing module (if available)."""
    if not HAS_GLORE:
        return None
    from federated_glore import split_sites
    site_dfs = split_sites(train_df, n_sites=N_SITES, random_state=SEED, stratify_col=LABEL_COL)
    glore_sites = [
        GLORESite(i, s.reset_index(drop=True), FEATURE_COLS, LABEL_COL)
        for i, s in enumerate(site_dfs)
    ]
    model = GLOREModel().fit(glore_sites)
    y_prob = model.predict_proba(test_df[FEATURE_COLS])
    y_true = test_df[LABEL_COL].to_numpy()
    return compute_metrics(y_true, y_prob)


# ── Evaluation ────────────────────────────────────────────────────────────

def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10,
) -> float:
    edges = np.linspace(0, 1, n_bins + 1)
    ece, n = 0.0, len(y_true)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi) if hi < 1 else (y_prob >= lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_prob = np.clip(y_prob, 1e-7, 1 - 1e-7)
    y_pred = (y_prob >= 0.5).astype(int)
    try:
        auroc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auroc = float("nan")
    return {
        "AUROC": auroc,
        "Brier": float(brier_score_loss(y_true, y_prob)),
        "ECE": expected_calibration_error(y_true, y_prob),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "LogLoss": float(log_loss(y_true, y_prob)),
    }


# ── Main pipeline ────────────────────────────────────────────────────────

def main() -> None:
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 72)
    print("  Federated Learning Pipeline - CAD Polygenic Risk Prediction")
    print("=" * 72)

    df = load_data()
    print(f"\nDataset : {MULTI_PRS_PATH.name}  ({len(df)} samples)")
    print(f"Features: {FEATURE_COLS}")
    print(f"Label   : {LABEL_COL}  {dict(df[LABEL_COL].value_counts())}")

    train_df, test_df = split_train_test(df)
    X_tr = train_df[FEATURE_COLS].values.astype(np.float64)
    y_tr = train_df[LABEL_COL].values.astype(np.float64)
    X_te = test_df[FEATURE_COLS].values.astype(np.float64)
    y_te = test_df[LABEL_COL].values.astype(np.float64)

    print(f"Train {len(train_df)} / Test {len(test_df)}")

    all_results = {}

    splits = [
        ("IID", lambda d: iid_split(d)),
        ("Non-IID (label α=0.5)", lambda d: non_iid_label_skew(d, alpha=0.5)),
        ("Non-IID (feature shift)", lambda d: non_iid_feature_shift(d)),
    ]

    for split_name, split_fn in splits:
        print(f"\n{'-' * 72}")
        print(f"  Data partition: {split_name}")
        print(f"{'-' * 72}")

        site_dfs = split_fn(train_df)
        site_data: SiteData = [
            (s[FEATURE_COLS].values.astype(np.float64),
             s[LABEL_COL].values.astype(np.float64))
            for s in site_dfs
        ]
        for i, s in enumerate(site_dfs):
            print(f"  Site {i+1}: n={len(s):>5}  pos_rate={s[LABEL_COL].mean():.3f}")

        split_res: dict = {}

        for kind in ("logreg", "mlp"):
            tag = kind.upper()
            print(f"\n  >> Model: {tag}")

            # local baselines
            loc = run_local(site_data, X_te, y_te, kind)
            loc_time = loc[0].get("total_time_sec", 0)
            for r in loc:
                print(f"      Local Site {r['site']}: AUROC={r['AUROC']:.4f}  Brier={r['Brier']:.4f}  ({r['site_time_sec']:.2f}s)")
            print(f"      Local total runtime: {loc_time:.2f}s")
            split_res[f"local_{kind}"] = loc

            # centralized
            cen = run_centralized(X_tr, y_tr, X_te, y_te, kind)
            print(f"      Centralized  : AUROC={cen['AUROC']:.4f}  Brier={cen['Brier']:.4f}  ({cen['total_time_sec']:.2f}s)")
            split_res[f"central_{kind}"] = cen

            # FedAvg (save full history)
            _, h_fa = run_fedavg(site_data, X_te, y_te, kind)
            fa = h_fa[-1]
            print(f"      FedAvg       : AUROC={fa['AUROC']:.4f}  Brier={fa['Brier']:.4f}  ({fa['total_time_sec']:.2f}s)")
            split_res[f"fedavg_{kind}"] = fa
            split_res[f"fedavg_{kind}_history"] = h_fa

            # FedProx (save full history)
            _, h_fp = run_fedprox(site_data, X_te, y_te, kind)
            fp = h_fp[-1]
            print(f"      FedProx(mu={FEDPROX_MU}): AUROC={fp['AUROC']:.4f}  Brier={fp['Brier']:.4f}  ({fp['total_time_sec']:.2f}s)")
            split_res[f"fedprox_{kind}"] = fp
            split_res[f"fedprox_{kind}_history"] = h_fp

            # FedAvg + DP (save full history)
            _, h_dp = run_fedavg(site_data, X_te, y_te, kind, use_dp=True)
            dp = h_dp[-1]
            eps = estimate_epsilon(DP_NOISE_MULT, FL_ROUNDS)
            print(f"      FedAvg+DP(eps~{eps:.1f}): AUROC={dp['AUROC']:.4f}  Brier={dp['Brier']:.4f}  ({dp['total_time_sec']:.2f}s)")
            split_res[f"fedavg_dp_{kind}"] = {**dp, "epsilon": eps}
            split_res[f"fedavg_dp_{kind}_history"] = h_dp

            print(f"\n      {'-'*58}")
            print(f"      Local vs Global ({tag}) -- per-site AUROC")
            print(f"      {'Site':<8} {'Local':>10} {'FedAvg':>10} {'FedProx':>10} {'D(FA-Loc)':>10}")
            print(f"      {'-'*58}")
            for site_r in loc:
                sid = site_r["site"]
                l_auc = site_r["AUROC"]
                fa_auc = fa["AUROC"]
                fp_auc = fp["AUROC"]
                delta = fa_auc - l_auc
                print(f"      Site {sid:<4} {l_auc:>10.4f} {fa_auc:>10.4f} {fp_auc:>10.4f} {delta:>+10.4f}")
            avg_local = np.mean([r["AUROC"] for r in loc])
            print(f"      {'Avg':<8} {avg_local:>10.4f} {fa['AUROC']:>10.4f} {fp['AUROC']:>10.4f} {fa['AUROC']-avg_local:>+10.4f}")

            checkpoints = [1, 5, 10, 15, 20, 25, 30]
            checkpoints = [c for c in checkpoints if c <= len(h_fa)]
            print(f"\n      {'-'*62}")
            print(f"      Global model convergence ({tag}) -- AUROC per round")
            print(f"      {'Round':<8} {'FedAvg':>10} {'FedProx':>10} {'FedAvg+DP':>10} {'Centralized':>12}")
            print(f"      {'-'*62}")
            for c in checkpoints:
                fa_r = h_fa[c - 1]["AUROC"]
                fp_r = h_fp[c - 1]["AUROC"]
                dp_r = h_dp[c - 1]["AUROC"]
                cen_str = f"{cen['AUROC']:.4f}" if c == checkpoints[-1] else "-"
                print(f"      {c:<8} {fa_r:>10.4f} {fp_r:>10.4f} {dp_r:>10.4f} {cen_str:>12}")

        # GLORE (only for the IID split, as reference)
        if split_name.startswith("IID"):
            glore_res = run_glore(train_df, test_df)
            if glore_res is not None:
                print(f"\n  >> GLORE (exact federated LR)")
                print(f"      AUROC={glore_res['AUROC']:.4f}  Brier={glore_res['Brier']:.4f}")
                split_res["glore"] = glore_res

        all_results[split_name] = split_res

    # ── save results ──
    out_path = OUTPUT_DIR / "fl_comparison_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[OK] Results saved to {out_path}")

    # ── summary table ──
    print(f"\n{'=' * 72}")
    print("  SUMMARY  (IID partition)")
    print(f"{'=' * 72}")
    iid = all_results.get("IID", {})
    rows = []
    for key, name in [
        ("local_mlp",      "Local (avg)"),
        ("central_mlp",    "Centralized"),
        ("fedavg_logreg",  "FedAvg-LR"),
        ("fedavg_mlp",     "FedAvg-MLP"),
        ("fedprox_mlp",    "FedProx-MLP"),
        ("fedavg_dp_mlp",  "FedAvg+DP-MLP"),
        ("glore",          "GLORE"),
    ]:
        v = iid.get(key)
        if v is None:
            continue
        if isinstance(v, list):
            avg = {m: np.mean([r[m] for r in v]) for m in ("AUROC", "Brier", "ECE", "Accuracy")}
            rt = v[0].get("total_time_sec", float("nan"))
        else:
            avg = v
            rt = v.get("total_time_sec", float("nan"))
        rows.append({"Method": name,
                      **{m: avg.get(m, float("nan")) for m in ("AUROC", "Brier", "ECE", "Accuracy")},
                      "Runtime(s)": rt})

    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False, float_format="%.4f"))

    # ── Local vs Global detailed comparison across all splits ──
    print(f"\n{'=' * 72}")
    print("  LOCAL vs GLOBAL MODEL COMPARISON  (MLP)")
    print(f"{'=' * 72}")
    for sn in all_results:
        sr = all_results[sn]
        loc_mlp = sr.get("local_mlp", [])
        fa_mlp = sr.get("fedavg_mlp", {})
        fp_mlp = sr.get("fedprox_mlp", {})
        cen_mlp = sr.get("central_mlp", {})
        if not loc_mlp:
            continue
        print(f"\n  [{sn}]")
        print(f"  {'':12} {'AUROC':>8} {'Brier':>8} {'ECE':>8} {'Accuracy':>10}")
        print(f"  {'-'*50}")
        for r in loc_mlp:
            print(f"  {'Local S'+str(r['site']):12} {r['AUROC']:>8.4f} {r['Brier']:>8.4f} "
                  f"{r['ECE']:>8.4f} {r['Accuracy']:>10.4f}")
        avg_l = {m: np.mean([r[m] for r in loc_mlp]) for m in ("AUROC", "Brier", "ECE", "Accuracy")}
        print(f"  {'Local Avg':12} {avg_l['AUROC']:>8.4f} {avg_l['Brier']:>8.4f} "
              f"{avg_l['ECE']:>8.4f} {avg_l['Accuracy']:>10.4f}")
        print(f"  {'-'*50}")
        for label, d in [("FedAvg", fa_mlp), ("FedProx", fp_mlp), ("Centralized", cen_mlp)]:
            if d:
                print(f"  {label:12} {d.get('AUROC',0):>8.4f} {d.get('Brier',0):>8.4f} "
                      f"{d.get('ECE',0):>8.4f} {d.get('Accuracy',0):>10.4f}")
        print(f"  {'-'*50}")
        if fa_mlp:
            gain = fa_mlp.get("AUROC", 0) - avg_l["AUROC"]
            print(f"  FedAvg vs Local Avg:  ΔAUROC = {gain:+.4f}")

    # ── Non-IID comparison ──
    print(f"\n{'=' * 72}")
    print("  NON-IID COMPARISON  (FedAvg-MLP vs FedProx-MLP)")
    print(f"{'=' * 72}")
    noniid_rows = []
    for sn in all_results:
        fa = all_results[sn].get("fedavg_mlp", {})
        fp = all_results[sn].get("fedprox_mlp", {})
        if fa and fp:
            noniid_rows.append({
                "Split": sn,
                "FedAvg AUROC": fa.get("AUROC", float("nan")),
                "FedProx AUROC": fp.get("AUROC", float("nan")),
                "Delta AUROC": fp.get("AUROC", 0) - fa.get("AUROC", 0),
            })
    if noniid_rows:
        print(pd.DataFrame(noniid_rows).to_string(index=False, float_format="%.4f"))

    # ── Per-round convergence summary (IID, MLP) ──
    print(f"\n{'=' * 72}")
    print("  CONVERGENCE TRACKING  (IID, MLP — AUROC per round)")
    print(f"{'=' * 72}")
    fa_hist = iid.get("fedavg_mlp_history", [])
    fp_hist = iid.get("fedprox_mlp_history", [])
    dp_hist = iid.get("fedavg_dp_mlp_history", [])
    if fa_hist:
        print(f"  {'Round':>6} {'FedAvg':>10} {'FedProx':>10} {'FedAvg+DP':>10}")
        print(f"  {'-'*40}")
        for r_idx in range(len(fa_hist)):
            rnd = fa_hist[r_idx]["round"]
            fa_a = fa_hist[r_idx]["AUROC"]
            fp_a = fp_hist[r_idx]["AUROC"] if r_idx < len(fp_hist) else float("nan")
            dp_a = dp_hist[r_idx]["AUROC"] if r_idx < len(dp_hist) else float("nan")
            print(f"  {rnd:>6} {fa_a:>10.4f} {fp_a:>10.4f} {dp_a:>10.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
