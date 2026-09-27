# Business Entity Resolution: reproducible pipeline

Given S1 (reference) and S2/S3 (noisy) business records, predict for every S1 entity which S2/S3
records refer to the same business. Metric: macro F0.5 over S1 entities, singletons included.

## Layout

```
code/business_entity_resolution/
├── README.md, requirements.txt
├── tests/test_normalize.py      # parser checks on real examples from the data
└── src/
    ├── config.py         paths (ER_DATA / ER_WORK / ER_OUT env overrides), seeds, K values
    ├── io_utils.py       TSV -> parquet, integer ids, ground-truth pairs
    ├── splits.py         hidden set H (20% of train S1) + 5 GBDT folds over the query set Q
    ├── translit.py       Indic -> Latin: token dictionary learnt from train pairs + ITRANS fallback
    ├── normalize.py      names (junk prefixes, aliases, domains, leet, legal suffixes) and addresses
    │                     (component parser: state/region, locality, postcode, house number, unit, street)
    ├── blocking.py       sparse TF-IDF top-k per (country, state): name / name+address / address-only,
    │                     reverse pass, exact keys; union with per-blocker scores and ranks
    ├── features.py       ~75 pairwise features (numba set overlaps, rapidfuzz, house numbers, name frequency)
    ├── gbdt.py           LightGBM GroupKFold helper (out-of-fold predictions)
    ├── prune.py          streaming prune model -> top-20 per S1  (= candidate_pairs.tsv)
    ├── stage1.py         pairwise LightGBM on the pruned set
    ├── context.py        context features (within-S1, competition, S2/S3 agreement, twins, CE)
    │                     -> stage-2 LightGBM -> isotonic -> decision tuning
    ├── decide.py         one-to-one assignment + per-S1 expected-F0.5 subset (numba Monte-Carlo)
    ├── evaluate.py       exact macro F0.5 + breakdowns
    ├── crossencoder.py   optional GPU cross-encoder: export / train / infer / France pseudo-labels
    ├── write_output.py   output TSVs + official validator
    └── run_pipeline.py   orchestrator
```

## Environment

CPU side (tested on an Apple M3 Max, 48 GB RAM, Python 3.11):
```bash
brew install libomp                     # macOS only, needed by LightGBM
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r requirements.txt
```
Peak memory is about 30 GB. Any Linux box with ≥48 GB of RAM works the same way.

GPU side (cross-encoder only; tested design target: 1× A10G 24 GB): `torch` (CUDA build),
`transformers`, `polars`, `pyarrow`, `scikit-learn`.

## Run

Data is expected at `<repo>/dataset/{train,test}/*.tsv` (override with `ER_DATA`). Intermediate files
go to `<repo>/work/` (`ER_WORK`) and outputs to `<repo>/output/` (`ER_OUT`).

```bash
cd code/business_entity_resolution/src
../../../.venv/bin/python run_pipeline.py          # full CPU pipeline -> output/*.tsv (validated)
```
Stages can be resumed: `run_pipeline.py --from prune`. Measured wall-clock times on the M3 Max:
io 10 s, normalize 2 min, sets 1.5 min, blocking 28 min, prune about 100 min (about 60 min with the per-fold scoring
now in `prune.py`), stage1 58 min, context + decision tuning 19 min, output 20 s. Total is about 3.5 hours.

Results (out-of-fold on 1.77M held-out train S1, 20% of S1 hidden as orphans): macro F0.5 **0.98654**
(US 0.98731, India 0.98537, singletons 0.98733). The test output passes `validate_submission.py --check-ids`.

### Optional cross-encoder (adds the `ce*` features to the stage-2 model)
```bash
python crossencoder.py export                                  # Mac: work/ce/*.parquet
# copy work/ce/ to the GPU box, then there:
python crossencoder.py train --model intfloat/multilingual-e5-small --out ce_e5s
python crossencoder.py infer --model ce_e5s --band band_train.parquet --out ce_train.parquet
python crossencoder.py infer --model ce_e5s --band band_test.parquet  --out ce_test.parquet
# copy ce_train.parquet / ce_test.parquet into work/, then on the Mac:
python run_pipeline.py --from context
```
France pseudo-labels (after a first full run): `python crossencoder.py pseudo`. Then fine-tune on the GPU with
`train --model ce_e5s --extra pseudo_fr.parquet --epochs 1 --out ce_e5s_fr`, re-infer, and rerun `--from context`.

The whole round trip is scripted in `gpu_steps.sh` (`GPU=user@host bash gpu_steps.sh`).

### Leaderboard probes
`write_output.py --params probe.json --tag NAME` writes `output/matching_results_NAME.tsv` using
per-country decision parameters, for example
`{"default": {"method": "ef", "a": 1.0, "b": 0.0, "delta": 0.1}, "France": {"b": -0.5}}`.
`python make_probes.py France` writes a ready-made grid of France-only variants.

## Models and licences
LightGBM (MIT). Optional cross-encoder: `intfloat/multilingual-e5-small` (MIT, 118M parameters).
Transliteration fallback: `indic-transliteration` (MIT). No model exceeds 8B parameters.
