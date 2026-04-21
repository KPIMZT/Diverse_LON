"""
msg_cor_analyze.py
============
Analyze correlation between algorithms performances and LONs features.
"""

import os
import numpy as np
import pandas as pd
import torch
from typing import List, Tuple


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from scipy.stats import spearmanr
from cor_viz import plot_feature_grid


# correlation analysis
def compute_spearman(
    df:       pd.DataFrame,
    features: List[str],
    metrics:  List[str],
) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        row: dict[str, str | float] = {"metric": metric}
        y   = df[metric].values
        for feat in features:
            x    = df[feat].values
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() < 3:
                row[feat] = np.nan
            else:
                rho, _ = spearmanr(x[mask], y[mask])
                row[feat] = round(float(rho), 4)
        rows.append(row)
    return pd.DataFrame(rows).set_index("metric")


def compute_rf(
    df:         pd.DataFrame,
    features:   List[str],
    metrics:    List[str],
    n_estimators: int,
    cv:           int,
    seed:         int,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    kf = KFold(n_splits=cv, shuffle=True, random_state=seed)
    r2_rows  = []
    imp_rows = []
    models   = {}
    for metric in metrics:
        y = df[metric].values
        X = df[features].values
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        if mask.sum() < cv + 1:
            r2_rows.append({"metric": metric, "rf_r2_cv": np.nan})
            imp_rows.append({"metric": metric, **{f: np.nan for f in features}})
            models[metric] = None
            continue
        Xm, ym = X[mask], y[mask]
        rf = RandomForestRegressor(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
        cv_scores = cross_val_score(rf, Xm, ym, cv=kf, scoring="r2")
        rf_r2_cv  = float(np.mean(cv_scores))
        rf.fit(Xm, ym)
        models[metric] = rf
    df_r2_cv = pd.DataFrame(r2_rows).set_index("metric")
    return df_r2_cv, models

def run_correlation(
        ns_path: str,
        bench_path: str,
        dim: int,
        features: List[str],
        alg_name: List[str],
        metrics: List[str],
        n_estimators: int,
        cv: int,
        seed: int,
        bins: int,
        save_path = "./results_cor_reg"
) -> None:
    os.makedirs(save_path, exist_ok=True)
    df_perf = pd.read_csv(bench_path)
    print(f"load: {bench_path}  ({len(df_perf)} rows, cols={list(df_perf.columns)})")
    data = torch.load(ns_path, weights_only=False)
    idx = [i for i, m in enumerate(data["_records_meta"]) if m["population"]=="novelty"]
    lon_records = data["_records_all_features"][idx]
    df_lon = pd.DataFrame(lon_records, columns=features)
    print(f"LON features: {len(df_lon)} rows, cols={list(df_lon.columns)}")
    df = pd.concat([df_perf, df_lon], axis=1)
    print(f"merged: {len(df)} rows")
    print(f"merged df: {len(df)} rows")
    print(f"cols: {list(df.columns)}\n")
    df.to_csv(f"{save_path}/dim{dim}_seed{seed}_{alg_name}.csv")
    spearman_all: dict = {}
    rf_r2_all:    dict = {}
    for alg in alg_name:
        alg_metrics = {m: f"{alg}_{m}" for m in metrics}
        sub = df[features + list(alg_metrics.values())].copy()
        sub = sub.rename(columns={v: k for k, v in alg_metrics.items()})
        df_sp = compute_spearman(sub, features, metrics)
        df_rf_r2,  models = compute_rf(sub, features, metrics, n_estimators, cv, seed)
        for metric in metrics:
            model = models[metric]
            if model is None:
                continue
            y = sub[metric].values
            X = sub[features].values
            mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            r2_cv = float(df_rf_r2.loc[metric, "rf_r2_cv"]) if metric in df_rf_r2.index else None



        feature_csv = f"{save_path}/dim{dim}_seed{seed}_{alg_name}.csv"
        plot_feature_grid(
            bench_path=bench_path,
            feature_path=feature_csv,
            feature_x="num_nodes",
            feature_y="global_funnel_size",
            alg_name=alg,
            metric="success_rate",
            n_bins=bins,
            show_values=False,
            save_path=f"{save_path}/dim{dim}_seed{seed}_fgrid_{alg}_{metric}.pdf",
        )

        spearman_all[alg] = df_sp
        rf_r2_all[alg]    = df_rf_r2

        print(f"━━━ {alg} ━━━")
        print("Spearman ρ:")
        print(df_sp.to_string())
        print("\nRF R² (CV):")
        print(df_rf_r2.to_string())


    for data_dict, label in [(spearman_all, "spearman"),  (rf_r2_all, "rf_r2_all")]:
        rows = []
        for alg, df_ in data_dict.items():
            for metric in df_.index:
                row = {"algorithm": alg, "metric": metric}
                row.update(dict(df_.loc[metric]))
                rows.append(row)
        out_path = os.path.join(save_path, f"dim{dim}_seed{seed}_{alg}_{label}.csv")
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"saved: {out_path}")




