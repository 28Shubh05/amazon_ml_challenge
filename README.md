# Amazon ML Challenge 2026: Business Entity Resolution

[![Evaluated Score](https://img.shields.io/badge/Leaderboard%20Macro%20F0.5-0.974-brightgreen)](#results)
[![Platform](https://img.shields.io/badge/OS-Ubuntu%20Linux%2024.04%20LTS-orange)](#hardware--compute-environment)
[![Hardware](https://img.shields.io/badge/Hardware-8%20vCPUs%20|%2064GB%20RAM%20|%20400GB%20SSD-blue)](#hardware--compute-environment)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

An end-to-end, high-precision machine learning solution for large-scale **Business Entity Resolution (ER)** across independent, noisy data sources for the Amazon ML Challenge 2026.

Given deduplicated reference entities from **Source 1**, the pipeline resolves all matching entity records from **Source 2** and **Source 3**, effectively addressing name variations, missing address components, multi-script Indic transliterations, and zero-shot geographic transfer (France).

---

## Final Results

* **Evaluated Leaderboard Macro $F_{0.5}$ Score:** **0.974**
* **Out-of-Fold Validation Performance (1,765,506 Source 1 entities)**:
  * **Macro $F_{0.5}$**: **0.98654**
  * **Micro Precision**: **0.9977** (99.77%)
  * **Micro Recall**: **0.9644** (96.44%)
  * **United States**: 0.98732
  * **India**: 0.98538
  * **Matched Entities**: 0.98649
  * **Singletons (Empty matches)**: 0.98741
* **Submission Deliverables**:
  * [`output/matching_results.tsv`](file:///home/sumitchint_work/amazonMl/output/matching_results.tsv) (91 MB, 1,732,544 rows, 5,634,283 matches)
  * [`output/candidate_pairs.tsv`](file:///home/sumitchint_work/amazonMl/output/candidate_pairs.tsv) (211 MB, 1,732,544 rows, 15,402,735 candidate pairs)
  * Official Validator: **`PASS — no blocking issues found. Safe to submit.`**

---

## Hardware & Compute Environment

The entire pipeline was trained and evaluated on a Cloud Virtual Machine with the following specifications:

* **Operating System:** Ubuntu Linux 24.04 LTS (x86_64)
* **Compute:** 8 vCPUs (4 physical cores with hyperthreading)
* **Memory:** 64 GB RAM
* **Storage:** 400 GB NVMe SSD
* **Python Runtime:** Python 3.12 (using `.venv` with `spawn` multiprocessing mode)

---

## Repository Structure

```
├── Documentation_template.md       # Official solution methodology writeup
├── README.md                       # Project overview & reproduction guide
├── make_submission_zip.sh          # Helper to package final submission zip
├── requirements.txt                # Python environment requirements
├── utils/
│   └── validate_submission.py      # Official competition validator
├── output/
│   ├── matching_results.tsv        # Final entity matches (leaderboard submission)
│   └── candidate_pairs.tsv         # Pruned candidate set fed to matching models
└── code/
    └── business_entity_resolution/
        ├── README.md               # Detailed technical documentation
        ├── requirements.txt        # Pinned dependencies
        ├── tests/
        │   └── test_normalize.py   # Unit tests for text & address normalization
        └── src/
            ├── config.py           # Paths, seeds, and hyperparameters
            ├── io_utils.py         # TSV to parquet conversion & integer ID indexing
            ├── splits.py           # 20% orphan hidden set + 5 GBDT validation folds
            ├── translit.py         # Indic -> Latin bilingual transliteration engine
            ├── normalize.py        # Name & address component parser
            ├── blocking.py         # Multi-channel sparse TF-IDF & exact key blocking
            ├── features.py         # 93 pairwise feature extraction engine (Numba/RapidFuzz)
            ├── gbdt.py             # LightGBM GroupKFold CV trainer
            ├── prune.py            # Streaming candidate reduction to top-20 per S1
            ├── stage1.py           # Primary pairwise matching model
            ├── context.py          # Agreement, competition, twin features & Stage-2 GBDT
            ├── decide.py           # 1-to-1 matching + Monte Carlo expected-F0.5 selection
            ├── evaluate.py         # Official macro F0.5 evaluation implementation
            └── run_pipeline.py     # End-to-end execution orchestrator
```

---

## Methodology Highlights

### 1. Robust Normalization & Transliteration
* **Indic Scripts**: Learned bilingual token dictionaries from training ground truth pairs with `indic-transliteration` ITRANS fallback and consonant skeleton snapping.
* **Address Parsing**: Component extraction isolating house numbers (`a_hnd`), unit designations (`a_unit`), street names (`a_street`), locality (`a_loc`), city (`a_city`), state (`a_state`), and postal codes (`a_pc`).
* **Legal Suffix Canonicalization**: Standardizes `Pvt Ltd`, `Corp`, `Inc`, `SARL`, `SAS`, `SCI` across international standards.
* **Domain Base Segmentation**: Viterbi word segmentation on trade URLs (e.g., `pclmedicalcentre.com` $\to$ `pcl medical centre`).

### 2. Multi-Channel Candidate Generation (Blocking)
* Sparse char 3-gram TF-IDF on normalized name (top 30 per entity).
* Sparse char 3-gram TF-IDF on name + street + locality (top 30 per entity).
* Word-level TF-IDF on address tokens and numeric components (top 30 per entity).
* Reverse pass for pool records with missing address details.
* Exact multi-attribute keys: `(country, locality, street)`, `(country, state, name_key)`, `(country, locality, house_number, street_token)`.
* Generates ~212M test pairs, retaining **98.63% recall ceiling** on held-out validation.

### 3. Two-Tier Gradient Boosted Matching Funnel
* **Stage 0 (Prune)**: Cheap-feature streaming LightGBM narrows the 212M pairs down to the top-20 per entity (15.4M pairs in `candidate_pairs.tsv`).
* **Stage 1 (Pairwise Matching)**: 93 pairwise features computed over string distances (RapidFuzz), Numba-accelerated token set overlaps, house number differences, and co-location frequency metrics, trained using 5-fold GroupKFold by entity.
* **Stage 2 (Contextual Modeling)**: 115 features incorporating S2/S3 cross-source agreement, within-S1 ranking/margin gaps, competition across entities (enforcing 1-to-1 matching), and twin entity detection.

### 4. Precision-Weighted Decision Engine
* **Metric Alignment**: Submissions are evaluated using macro $F_{0.5}$, where false merges penalize score $2\times$ more than false negatives.
* **Isotonic Probability Calibration**: Non-parametric calibration of raw model margins into posterior probabilities.
* **Monte Carlo Optimization**: Entity-by-entity simulation sampling binary cluster outcomes to select the exact candidate subset maximizing expected macro $F_{0.5}$.

---

## Reproduction Instructions

### 1. Setup Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r code/business_entity_resolution/requirements.txt
```

### 2. Run Pipeline End-to-End
```bash
cd code/business_entity_resolution/src
ER_JOBS=8 OMP_NUM_THREADS=8 ../../../.venv/bin/python run_pipeline.py
```

### 3. Validate Submission Outputs
```bash
cd /home/sumitchint_work/amazonMl
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test \
    --check-ids
```

### 4. Build Final Submission Zip
```bash
bash make_submission_zip.sh 28Shubh05
```
This generates `28Shubh05_submission.zip` matching the required directory format.
