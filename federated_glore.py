from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -35, 35)
    return 1.0 / (1.0 + np.exp(-x))


def add_intercept(x: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(x)), x])


def split_sites(
    df: pd.DataFrame,
    n_sites: int = 3,
    random_state: int = 88,
    stratify_col: str = "label",
) -> List[pd.DataFrame]:
    rng = np.random.default_rng(random_state)
    shuffled = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    if stratify_col not in shuffled.columns:
        return [chunk.reset_index(drop=True) for chunk in np.array_split(shuffled, n_sites)]

    sites = [[] for _ in range(n_sites)]
    for _, group in shuffled.groupby(stratify_col):
        idx = rng.permutation(len(group))
        parts = np.array_split(group.iloc[idx], n_sites)
        for site_id, part in enumerate(parts):
            sites[site_id].append(part)

    return [pd.concat(parts).sample(frac=1.0, random_state=random_state).reset_index(drop=True) for parts in sites]


@dataclass
class LocalSite:
    site_id: int
    data: pd.DataFrame
    feature_cols: Sequence[str]
    label_col: str = "label"

    def __post_init__(self) -> None:
        self.x = add_intercept(self.data.loc[:, self.feature_cols].to_numpy(dtype=float))
        self.y = self.data.loc[:, self.label_col].to_numpy(dtype=float)

    def local_statistics(self, beta: np.ndarray, l2: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        eta = self.x @ beta
        p = sigmoid(eta)
        w = np.clip(p * (1.0 - p), 1e-6, None)
        z = eta + (self.y - p) / w

        wx = self.x * w[:, None]
        xtwx = self.x.T @ wx
        xtwz = self.x.T @ (w * z)

        if l2 > 0:
            penalty = np.eye(self.x.shape[1]) * l2
            penalty[0, 0] = 0.0
            xtwx = xtwx + penalty

        return xtwx, xtwz

    def predict_proba(self, beta: np.ndarray) -> np.ndarray:
        return sigmoid(self.x @ beta)


@dataclass
class FederatedLogisticRegression:
    max_iter: int = 50
    tol: float = 1e-6
    l2: float = 1e-6

    def fit(self, sites: Sequence[LocalSite]) -> "FederatedLogisticRegression":
        n_features = sites[0].x.shape[1]
        beta = np.zeros(n_features, dtype=float)
        history = []

        for iteration in range(1, self.max_iter + 1):
            xtwx_total = np.zeros((n_features, n_features), dtype=float)
            xtwz_total = np.zeros(n_features, dtype=float)

            for site in sites:
                xtwx_i, xtwz_i = site.local_statistics(beta, l2=self.l2)
                xtwx_total += xtwx_i
                xtwz_total += xtwz_i

            beta_new = np.linalg.solve(xtwx_total, xtwz_total)
            step_norm = np.linalg.norm(beta_new - beta)
            beta = beta_new

            y_true_all = np.concatenate([site.y for site in sites])
            y_prob_all = np.concatenate([site.predict_proba(beta) for site in sites])
            loss = log_loss(y_true_all, y_prob_all)

            history.append(
                {
                    "iter": iteration,
                    "log_loss": loss,
                    "step_norm": step_norm,
                }
            )

            if step_norm < self.tol:
                break

        self.coef_ = beta[1:]
        self.intercept_ = beta[0]
        self.beta_ = beta
        self.history_ = pd.DataFrame(history)
        self.n_iter_ = len(history)
        return self

    def predict_proba(self, x: np.ndarray | pd.DataFrame) -> np.ndarray:
        if isinstance(x, pd.DataFrame):
            x = x.to_numpy(dtype=float)
        x_aug = add_intercept(np.asarray(x, dtype=float))
        return sigmoid(x_aug @ self.beta_)

    def predict(self, x: np.ndarray | pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(x) >= threshold).astype(int)


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "AUROC": roc_auc_score(y_true, y_prob),
        "Brier": brier_score_loss(y_true, y_prob),
        "Accuracy": accuracy_score(y_true, y_pred),
        "LogLoss": log_loss(y_true, y_prob),
    }


def calibration_table(
    y_true: Sequence[float],
    y_prob: Sequence[float],
    model_name: str,
    n_bins: int = 10,
) -> pd.DataFrame:
    df = pd.DataFrame({"y": y_true, "p": y_prob}).copy()
    df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")
    calib = (
        df.groupby("bin", observed=False)
        .agg(mean_pred=("p", "mean"), obs_rate=("y", "mean"), count=("y", "size"))
        .reset_index(drop=True)
    )
    calib["Model"] = model_name
    return calib
