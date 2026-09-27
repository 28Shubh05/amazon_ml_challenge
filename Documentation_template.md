# ML Challenge 2026: Business Entity Resolution Solution Documentation

**Team Name:** 28Shubh05  
**Team Members:** Shubh, Sumit  
**Submission Date:** September 27, 2026  
**Hardware Environment:** Cloud Virtual Machine (400 GB NVMe Storage, 8 vCPUs / 4 Cores, 64 GB RAM, Ubuntu 24.04 LTS)  

---

## 1. Executive Summary

This solution presents an end-to-end, high-precision Business Entity Resolution (ER) pipeline designed to resolve noisy, disparate business entity records from Sources 2 and 3 against reference entities in Source 1. The architecture implements a two-tier sparse TF-IDF and exact key candidate generation (blocking) stage, followed by streaming candidate pruning to top-20 candidates per entity. Pairwise match probabilities are predicted using a 93-feature LightGBM GBDT model, augmented by a Stage-2 context model capturing cross-source agreement, within-entity competition, and twin entity discrimination, followed by isotonic probability calibration and Monte Carlo expected-$F_{0.5}$ decision threshold tuning. The solution achieves an official macro $F_{0.5}$ score of **0.974**.

---

## 2. Methodology

### 2.1 Problem Analysis
During exploratory data analysis across the 26.4 million records, several core noise patterns and data characteristics were identified:
* **Lexical & Legal Variations**: High frequency of abbreviated legal suffixes (`Pvt Ltd`, `LLC`, `Corp`, `SARL`, `SAS`), DBA/trade aliases, concatenated domain names (`examplecompany.com`), and leetspeak substitution (`0` for `o`, `5` for `s`).
* **Multilingual & Transliteration Noise**: Indian business records frequently appear in Devanagari, Bengali, Tamil, Telugu, and other Indic scripts for records whose reference representation is in Latin English.
* **Address Heterogeneity**: Address formats differ sharply across jurisdictions:
  * *US*: Standard street numbering, directional indicators (`E`, `NW`), state abbreviations.
  * *India*: Landmark-relative descriptors (`Near SBI ATM`, `Opposite Market`), sector/block/plot numbering, municipal variations, and historical state name aliases (e.g., Telangana records cataloged under Andhra Pradesh).
  * *France*: European street numbering (`55bis`, `18 Rue`), department numbers, and regional groupings (`Hauts-de-France`, `Nouvelle-Aquitaine`).
* **Singleton Preponderance**: A substantial proportion of Source 1 entities have zero true matches in Sources 2 and 3. Under the competition macro $F_{0.5}$ metric, singletons receive a score of $1.0$ for predicting an empty list and $0.0$ for any false merge, making precision and false-merge avoidance paramount.

### 2.2 Solution Strategy
The pipeline follows a multi-stage funnel designed for scalability, zero data leakage, and maximum precision:

1. **Normalization & Canonicalization**:
   * Token-level bilingual transliteration for Indic scripts via train-learned translation dictionaries with ITRANS fallback.
   * Address parsing separating house number (`a_hnd`), unit (`a_unit`), street (`a_street`), locality (`a_loc`), city (`a_city`), state (`a_state`), and postal code (`a_pc`).
   * Domain base segmentation via unigram cost Viterbi parsing.
2. **Multi-Channel Candidate Generation (Blocking)**:
   * Sparse TF-IDF cosine top-$k$ indexing partitioned by `(country, state)` across three distinct text representations: name-only (char 3-gram), name + address (char 3-gram), and address tokens (word level).
   * Reverse pass mapping orphan/unaddressed pool records back to top Source 1 entities.
   * High-precision exact blocking keys across locality, street, and normalized name keys.
3. **Streaming Candidate Pruning**:
   * A cheap-feature LightGBM model trained on a downsampled union streams candidates and filters down to the top-20 candidate pairs per Source 1 entity ($K=20$, $p \ge 0.005$). This forms `candidate_pairs.tsv`.
4. **Primary Pairwise Matching (Stage 1)**:
   * 93 pairwise string, token, frequency, and address features trained using 5-fold GroupKFold cross-validation grouped by Source 1 entity.
5. **Contextual Refinement & Decision Tuning (Stage 2)**:
   * Higher-order context features including within-S1 ranking and margin gaps, cross-entity candidate competition (enforcing one-to-one mapping), S2/S3 cross-agreement, and twin detection.
   * Isotonic probability calibration mapping raw GBDT margins to true posterior probabilities.
   * Per-entity Monte Carlo simulation selecting the candidate subset that maximizes the expected $F_{0.5}$ score.

---

## 3. Candidate Generation (Blocking)

The blocking stage reduces the $O(N_1 \times (N_2 + N_3))$ search space (~20 trillion possible pairs) down to a high-recall candidate pool:

* **Blocking Keys & Channels**:
  * Sparse char 3-gram TF-IDF on normalized core name (top 30 per entity).
  * Sparse char 3-gram TF-IDF on name + street + locality (top 30 per entity).
  * Word-level TF-IDF on address tokens and numeric components (top 30 per entity).
  * Reverse pass on pool records with missing address fields (top 20 reverse matches).
  * Exact multi-attribute keys:
    * `(country, locality, street)`
    * `(country, state, name_key)`
    * `(country, locality, house_number, first_street_token)`
* **Candidate Pairs Generated**:
  * Train set: 225,727,340 raw pairs $\to$ **17,287,649 pruned pairs** ($7.83$ candidates/S1).
  * Test set: 211,901,856 raw pairs $\to$ **15,402,735 pruned pairs** (exported as [`output/candidate_pairs.tsv`](file:///home/sumitchint_work/amazonMl/output/candidate_pairs.tsv)).
* **Recall Retention**:
  * Evaluated on held-out validation queries ($Q$), candidate generation achieves **98.63% recall ceiling** at $K=20$ (Recall@5: 92.66%, Recall@10: 98.42%, Recall@15: 98.60%, Recall@20: 98.63%).

---

## 4. Matching Model

### 4.1 Feature Engineering (93 Base + 22 Context Features)
* **Name Similarity Features**:
  * RapidFuzz token set ratio, token sort ratio, partial ratio, full ratio, Jaro-Winkler, Levenshtein distance on concatenated names.
  * Consonant skeleton similarity (stripping vowels/y/h and deduplicating consonants).
  * Multi-variant maximum ratio evaluating DBA/alias and domain-segmented names.
  * Legal suffix compatibility categorization (exact match, missing on one side, incompatible).
* **Address Similarity Features**:
  * Numba CSR-accelerated token set metrics: Jaccard, containment ratios ($A \to B$ and $B \to A$), and IDF-weighted Jaccard.
  * Numeric house number comparison: exact match, prefix match, absolute difference, relative difference, Levenshtein edit distance on digit strings.
  * Street name Jaro-Winkler and token set ratio.
  * Exact categorical matching on parsed state, city, unit, and postal code.
* **Frequency & Ambiguity Features**:
  * Source 1 and pool frequency counts per name key, name key + city, and full address.
* **Context & Relational Features (Stage 2)**:
  * Within-S1 features: rank, probability gap to top candidate, probability share, count of high-confidence predictions ($p > 0.5$).
  * Competition features: margin between candidate's score for this S1 vs. best score for any other S1 (penalizes one-to-many false merges).
  * Cross-source agreement: similarity of Source 2 candidates to Source 3 candidates matching the same Source 1 entity, weighted by probability.
  * Twin detection: flags same-name entities at distinct house numbers to prevent merging multi-unit or branch businesses.

### 4.2 Model Architecture & Training
* **Model Type**: LightGBM Gradient Boosted Decision Trees (`LGBMClassifier`) with binary logloss objective.
  * Hyperparameters: `num_leaves=255`, `learning_rate=0.05`, `feature_fraction=0.8`, `bagging_fraction=0.8`, `lambda_l2=1.0`, `min_data_in_leaf=100`.
* **Validation Strategy**: 5-fold GroupKFold partitioned strictly by Source 1 entity ID, ensuring zero entity overlap between training and validation folds. A 20% slice of Source 1 entities is withheld completely as an orphan simulation set ($H$).
* **Probability Calibration**: Non-parametric `IsotonicRegression` fitted on out-of-fold Stage-2 margins to ensure well-calibrated posterior probabilities.

### 4.3 Decision Rule & Threshold Optimization
Rather than using a fixed global probability cutoff, the pipeline applies a two-step decision engine:
1. **Competitive Assignment**: Enforces 1-to-1 matching by retaining candidates only for their highest-scoring Source 1 entity (subject to competitive margin $\delta = 0.2$).
2. **Expected-$F_{0.5}$ Subset Selection**: For each Source 1 entity, a Numba-accelerated Monte Carlo simulation samples binary outcomes from the calibrated probabilities and selects the candidate subset size $k$ that maximizes expected entity $F_{0.5}$.

---

## 5. Results & Error Analysis

### 5.1 Quantitative Results
* **Official Evaluated Macro $F_{0.5}$ Score:** **0.974**
* **Validation Metrics (Out-of-Fold on 1,765,506 S1 Entities)**:
  * **Macro $F_{0.5}$**: **0.98654**
  * **Micro Precision**: **0.9977** (99.77%)
  * **Micro Recall**: **0.9644** (96.44%)
  * **United States**: 0.98732
  * **India**: 0.98538
  * **Matched Entities ($n = 1,667,137$)**: 0.98649
  * **Singletons ($n = 98,369$)**: 0.98741
* **Test Set Outputs**:
  * [`output/matching_results.tsv`](file:///home/sumitchint_work/amazonMl/output/matching_results.tsv): 5,634,283 total matches (1,628,115 matched entities, 104,429 empty singletons).
  * [`output/candidate_pairs.tsv`](file:///home/sumitchint_work/amazonMl/output/candidate_pairs.tsv): 15,402,735 candidate pairs.
  * **Validator Status**: `PASS` (verified against all 9,969,589 test Source-2/3 IDs).

### 5.2 Error Analysis
* **Common False Positives (Wrong Merges)**:
  * *Retail Chains & Franchises*: Businesses with identical chain names located in dense commercial centers where address street numbers are missing or generic.
  * *Co-located Businesses*: Different legal entities operating within the same commercial complex or office building where suite/unit numbers were omitted.
* **Common False Negatives (Missed Matches)**:
  * *Severe Address Truncation*: Pool records containing only a city name with no street, locality, or postal code.
  * *Severe Typographical Corruption*: Brand names altered by 3+ character transpositions or unmapped regional transliterations.

---

## 6. Conclusion

The solution demonstrates that combining fine-grained domain-specific normalization (Indic transliteration, address component parsing, legal suffix canonicalization) with multi-channel sparse candidate blocking and a two-stage gradient boosted tree framework yields high entity resolution accuracy. By explicitly optimizing for the precision-weighted macro $F_{0.5}$ metric through calibrated expected-utility subset selection, the pipeline minimizes costly false merges while retaining high recall across US, India, and zero-shot France test entities.

---

## Appendix

### A. Code Artifacts & Reproduction
* **Source Location**: All source code is self-contained in `code/business_entity_resolution/src/`.
* **Execution Script**:
  ```bash
  cd code/business_entity_resolution/src
  ER_JOBS=8 OMP_NUM_THREADS=8 ../../../.venv/bin/python run_pipeline.py
  ```
* **Pipeline Stages**:
  * `io_utils.py`: Converts TSVs to parquet caches.
  * `splits.py`: Generates the 20% orphan hidden set and 5 CV folds.
  * `normalize.py`: Normalizes names and addresses, builds transliteration mappings.
  * `features.py`: Computes CSR token set overlaps and pairwise features.
  * `blocking.py`: Multi-channel sparse TF-IDF and exact key candidate generation.
  * `prune.py`: Prunes candidate pairs down to top-20 per Source 1 entity.
  * `stage1.py`: Extracts full feature matrix and trains 5-fold LightGBM model.
  * `context.py`: Computes cross-source agreement/competition features, trains Stage-2 model, and calibrates probabilities.
  * `write_output.py`: Writes TSV outputs and runs the official submission validator.

### B. Hardware & Compute Specifications
* **Infrastructure**: Cloud Virtual Machine
* **Storage**: 400 GB NVMe SSD
* **Compute**: 8 vCPUs (4 physical cores with hyperthreading)
* **Memory**: 64 GB RAM
* **OS**: Ubuntu Linux 24.04 LTS (x86_64)
* **Execution Time**: ~3.5 hours end-to-end runtime.
