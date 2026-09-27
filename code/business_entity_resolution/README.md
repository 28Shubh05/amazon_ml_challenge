# Business Entity Resolution: Reproducible Pipeline

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

## Environment & Hardware

The pipeline is benchmarked and validated on the following Cloud Virtual Machine environment:
* **Operating System:** Ubuntu Linux 24.04 LTS (x86_64)
* **Compute:** 8 vCPUs (4 physical cores with hyperthreading)
* **Memory:** 64 GB RAM (peak pipeline memory usage is ~20–30 GB)
* **Storage:** 400 GB NVMe SSD
* **Python Runtime:** Python 3.12 (with `spawn` multiprocessing start method configured)

### Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Execution

Data is expected at `<repo>/dataset/{train,test}/*.tsv` (override with `ER_DATA`). Intermediate files
go to `<repo>/work/` (`ER_WORK`) and outputs to `<repo>/output/` (`ER_OUT`).

```bash
cd code/business_entity_resolution/src
ER_JOBS=8 OMP_NUM_THREADS=8 ../../../.venv/bin/python run_pipeline.py
```

Stages can be resumed if needed: `run_pipeline.py --from <stage>` (stages: `io`, `splits`, `normalize`, `sets`, `blocking`, `prune`, `stage1`, `context`, `output`).

### Measured Stage Runtimes (on 8 vCPU / 64 GB RAM VM):
* `io`: ~21 seconds (26.4M raw TSV records -> parquet caches)
* `splits`: <1 second (80/20 train query/orphan split + 5 folds)
* `normalize`: ~11.8 minutes (multiprocess parsing across 24M records)
* `sets`: ~1.4 minutes (token set sorting and CSR IDF dictionaries)
* `blocking`: ~78 minutes (sparse TF-IDF top-$k$ + exact keys across train and test)
* `prune`: ~4.8 hours (streaming candidate reduction down to 15.4M test pairs)
* `stage1`: ~3.2 hours (full 93 pairwise feature extraction + 5-fold LightGBM)
* `context`: ~1.4 hours (cross-agreement, competition features, Stage-2 GBDT & isotonic calibration)
* `output`: ~56 seconds (TSV exports + official submission validator check)

## Results

* **Leaderboard Evaluation Macro F0.5 Score:** **0.974**
* **Out-of-Fold Validation on 1.77M Source 1 entities:** Macro F0.5 **0.98654**
  * Precision: 0.9977
  * Recall: 0.9644
  * United States: 0.98732
  * India: 0.98538
  * Matched entities ($n=1,667,137$): 0.98649
  * Singletons ($n=98,369$): 0.98741
* **Submission Outputs:**
  * `output/matching_results.tsv`: 1,732,544 rows, 5,634,283 matches (104,429 singletons)
  * `output/candidate_pairs.tsv`: 1,732,544 rows, 15,402,735 candidate pairs
  * Verified against official validator: `PASS — no blocking issues found. Safe to submit.`

## Models and Licences
* **LightGBM:** MIT License
* **Transliteration:** `indic-transliteration` (MIT License)
* All models are strictly under 8 Billion parameters and adhere to competition fair-play rules (zero external lookup).
