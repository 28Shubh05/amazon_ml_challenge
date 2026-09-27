"""Stage C: cheap features on the blocking union + prune model -> pruned candidate set.

The union is ~100 candidates per S1 (200M+ pairs), so this stage streams:
  pass 1: cheap features on a training sample of Q (all positives + 10% of negatives, weight 10)
          -> 5 fold models (GroupKFold by S1)
  pass 2: stream the union in S1-aligned chunks, compute cheap features, score
          (Q rows: their own fold's model = out-of-fold; H / test: average of fold models),
          keep top PRUNE_K per S1 with p >= PRUNE_MIN_P.
The pruned set is exactly what every later model scores, i.e. candidate_pairs.tsv for test.
"""
import time

import numpy as np
import polars as pl

from config import N_FOLDS, N_JOBS, PRUNE_K, PRUNE_MIN_P, SEED, wpath
from features import Ctx, features, features_chunked
from gbdt import cv_train
from splits import load_splits

BLOCK_FEATS = ["s_name", "r_name", "s_na", "r_na", "s_ad", "r_ad", "s_rev", "x", "n_cand"]
NEG_RATE = 0.10


def load_union(split: str) -> pl.DataFrame:
    c = pl.read_parquet(wpath(f"cands_{split}.parquet"))
    return c.with_columns(pl.len().over("s1").cast(pl.Float32).alias("n_cand"),
                          pl.col("x").cast(pl.Float32), pl.col("r_name", "r_na", "r_ad").cast(pl.Float32))


def label(df: pl.DataFrame) -> pl.DataFrame:
    gt = pl.read_parquet(wpath("gt_pairs.parquet")).with_columns(pl.lit(1, pl.Int8).alias("y"))
    return df.join(gt, on=["s1", "cand"], how="left").with_columns(pl.col("y").fill_null(0))


def _feat_cols(df):
    return [c for c in df.columns if c not in ("s1", "cand", "y", "hidden", "fold", "w")]


def train_models():
    qids = load_splits().filter(~pl.col("hidden")).select(pl.col("idx").alias("s1"), "fold")
    q = label(load_union("train").join(qids, on="s1"))
    rng = np.random.default_rng(SEED)
    keep = (q["y"].to_numpy() == 1) | (rng.random(q.height) < NEG_RATE)
    sample = q.filter(pl.Series(keep)).with_columns(
        pl.when(pl.col("y") == 1).then(1.0).otherwise(1.0 / NEG_RATE).cast(pl.Float32).alias("w"))
    print(f"prune: Q union {q.height:,} rows, training sample {sample.height:,} ({sample['y'].sum():,} pos)")
    del q
    f = features_chunked(Ctx("train"), sample, "cheap")
    feats = _feat_cols(f)
    _, _, models, imp = cv_train(f, feats, params=dict(learning_rate=0.1, num_leaves=127), rounds=1000,
                                 weight_col="w", log="prune")
    print(imp.head(12))
    return models, feats


def score_split(split, models, feats, fold_of=None, chunk=6_000_000):
    u = load_union(split)
    ctx = Ctx(split)
    s1 = u["s1"].to_numpy()
    bounds = [0]
    while bounds[-1] < len(s1):
        j = min(bounds[-1] + chunk, len(s1))
        while j < len(s1) and s1[j] == s1[j - 1]:
            j += 1
        bounds.append(j)
    parts, t0 = [], time.time()
    for a, b in zip(bounds[:-1], bounds[1:]):
        c = features(ctx, u.slice(a, b - a), "cheap")
        X = c.select(feats).to_numpy().astype(np.float32)
        fo = fold_of[c["s1"].to_numpy()] if fold_of is not None else np.full(c.height, -1)
        p = np.zeros(c.height, np.float32)
        avg = fo == -1                       # H / test rows: average of all fold models
        for k, m in enumerate(models):
            rows = np.flatnonzero(avg | (fo == k))   # Q rows: only their own (out-of-fold) model
            if rows.size:
                pk = m.predict(X[rows], num_threads=N_JOBS)
                p[rows] += np.where(avg[rows], pk / N_FOLDS, pk)
        c = (c.select(["s1", "cand"] + BLOCK_FEATS).with_columns(pl.Series("p_prune", p))
              .with_columns(pl.col("p_prune").rank("ordinal", descending=True).over("s1").cast(pl.Float32)
                            .alias("r_prune"))
              .filter((pl.col("r_prune") <= PRUNE_K) & (pl.col("p_prune") >= PRUNE_MIN_P)))
        parts.append(c)
        print(f"  prune-score {split}: {b:,}/{len(s1):,} ({time.time() - t0:.0f}s)", flush=True)
    out = pl.concat(parts).sort("s1", "cand")
    out.write_parquet(wpath(f"pruned_{split}.parquet"))
    print(f"{split}: pruned to {out.height:,} pairs")


def main():
    t0 = time.time()
    models, feats = train_models()
    sp = load_splits()
    q = sp.filter(~pl.col("hidden"))
    fold_of = np.full(int(sp["idx"].max()) + 1, -1, np.int64)   # S1 idx -> fold (-1 = hidden)
    fold_of[q["idx"].to_numpy()] = q["fold"].to_numpy()
    score_split("train", models, feats, fold_of)
    score_split("test", models, feats)
    report()
    print(f"prune done in {time.time() - t0:.0f}s")


def report():
    gt = pl.read_parquet(wpath("gt_pairs.parquet"))
    sp = load_splits().rename({"idx": "s1"})
    q = sp.filter(~pl.col("hidden")).select("s1")
    p = pl.read_parquet(wpath("pruned_train.parquet"), columns=["s1", "cand", "r_prune"]).join(q, on="s1")
    g = gt.join(q, on="s1")
    hit = g.join(p, on=["s1", "cand"], how="left")
    for k in (5, 10, 15, 20):
        r = hit.filter(pl.col("r_prune") <= k).height / g.height
        print(f"  pruned recall@{k}: {r:.5f}")
    print(f"  pairs per Q S1: {p.height / q.height:.2f}")


if __name__ == "__main__":
    main()
