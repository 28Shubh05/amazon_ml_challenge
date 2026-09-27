"""LightGBM cross-validation helper shared by prune / stage-1 / stage-2.

GroupKFold by S1 via the precomputed `fold` column. Early stopping uses an inner group split of the
training folds (never the OOF fold). Easy negatives can be down-sampled with inverse-probability weights.
Memory-lean: rows are gathered once per fold and predictions are made in chunks.
"""
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from config import N_FOLDS, N_JOBS, SEED

BASE = dict(objective="binary", learning_rate=0.05, num_leaves=255, min_data_in_leaf=100,
            feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
            max_bin=255, num_threads=N_JOBS, verbose=-1, seed=SEED)


def _rows(df: pl.DataFrame, feats, idx=None) -> np.ndarray:
    d = df.select(feats)
    if idx is not None:
        d = d[idx]
    return d.to_numpy().astype(np.float32, copy=False)


def predict_chunked(model, df: pl.DataFrame, feats, idx=None, chunk=2_000_000) -> np.ndarray:
    idx = np.arange(df.height) if idx is None else idx
    out = np.empty(len(idx), np.float32)
    for i in range(0, len(idx), chunk):
        out[i:i + chunk] = model.predict(_rows(df, feats, idx[i:i + chunk]), num_threads=N_JOBS)
    return out


def cv_train(df: pl.DataFrame, feats, label="y", params=None, rounds=3000, neg_sample=None,
             predict=None, log="", weight_col=None):
    """df needs columns s1, fold, label, feats.  neg_sample: (bool expr for easy rows, keep_frac): easy
    negatives are down-sampled and re-weighted.  predict: dict name -> DataFrame scored with the fold average.
    Returns oof (aligned to df), preds dict, models, importance DataFrame."""
    p = dict(BASE, **(params or {}))
    oof = np.zeros(df.height, np.float32)
    preds = {k: np.zeros(v.height, np.float32) for k, v in (predict or {}).items()}
    models, imp = [], np.zeros(len(feats))
    fold = df["fold"].to_numpy()
    y_all = df[label].to_numpy()
    w_all = df[weight_col].to_numpy().astype(np.float32) if weight_col else np.ones(df.height, np.float32)
    easy = np.zeros(df.height, bool)
    if neg_sample is not None:
        easy = df.select(neg_sample[0].alias("e"))["e"].to_numpy() & (y_all == 0)
        frac = neg_sample[1]
    grp = (df["s1"].to_numpy().astype(np.int64) * 2654435761 % 1000) < 50   # inner early-stopping groups
    rng = np.random.default_rng(SEED)
    for k in range(N_FOLDS):
        t0 = time.time()
        keep = fold != k
        w = w_all.copy()
        if neg_sample is not None:
            keep &= ~easy | (rng.random(df.height) < frac)
            w = np.where(easy, w / frac, w).astype(np.float32)
        tr_idx = np.flatnonzero(keep & ~grp)
        es_idx = np.flatnonzero(keep & grp)
        dtr = lgb.Dataset(_rows(df, feats, tr_idx), y_all[tr_idx], weight=w[tr_idx], free_raw_data=True)
        dva = lgb.Dataset(_rows(df, feats, es_idx), y_all[es_idx], weight=w[es_idx], reference=dtr)
        m = lgb.train(p, dtr, rounds, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
        del dtr, dva
        va_idx = np.flatnonzero(fold == k)
        oof[va_idx] = predict_chunked(m, df, feats, va_idx)
        for name, frame in (predict or {}).items():
            preds[name] += predict_chunked(m, frame, feats) / N_FOLDS
        imp += m.feature_importance("gain")
        models.append(m)
        print(f"  {log} fold {k}: {m.best_iteration} it, train rows {tr_idx.size:,}, {time.time() - t0:.0f}s",
              flush=True)
    imp_df = pl.DataFrame({"feature": feats, "gain": imp / imp.sum()}).sort("gain", descending=True)
    return oof, preds, models, imp_df
