# Research Dataset Metrics

## 1. Dataset Cleaning Metrics

| Metric | Value |
|---|---:|
| Original total records | NA |
| Final total records | 195 |
| Removed records | NA |
| Retention rate | NA |
| Compression ratio | NA |
| Noise reduction rate | NA |
| Crawl-error records still present | 0 |

## 2. Refusal Metrics

| Metric | Value |
|---|---:|
| Total refusal-containing records | 9 |
| Answerable records (heuristic) | 134 |
| Over-refusal records | 0 |
| Over-Refusal Rate (ORR) | 0.0000 |

## 3. Diversity Metrics

| Metric | Value |
|---|---:|
| Normalized task entropy | 0.9797 |
| Linguistic Diversity Index (LDI) | 0.8785 |
| Source entropy (bits) | NA |
| Average assistant response length (words) | 41.9872 |
| Std assistant response length (words) | 93.9740 |
| Vocabulary size | 2619 |
| Total tokens | 21152 |
| Type-Token Ratio (TTR) | 0.1238 |
| Vendi score (effective rank) | 51.8376 |
| Normalized Vendi score | 0.1329 |
| Vendi backend | sentence-transformers:all-MiniLM-L6-v2 |
| Vendi sample size | 390 |

## 4. Redundancy Metrics

| Metric | Value |
|---|---:|
| Duplicate response groups | 37 |
| Duplicated response items | 97 |
| Duplicate response rate | 0.2487 |
| Self-BLEU mean | 0.1226 |
| Self-BLEU std | 0.1966 |
| Self-BLEU samples used | 300 |

## 5. Grounding Metrics

| Metric | Value |
|---|---:|
| Mean context-answer similarity | 0.5126 |
| Std context-answer similarity | 0.2632 |
| Context-answer similarity backend | sentence-transformers:all-MiniLM-L6-v2 |
| Context-answer pairs used | 195 |
| Extractive QA records evaluated | 42 |
| Extractive exact substring match count | 2 |
| Extractive grounding accuracy | 0.0476 |

## 6. Dependency Metrics

| Metric | Value |
|---|---:|
| Mutual information: task-language | 0.0353 |
| Mutual information: task-response length bin | 0.4022 |
| Mutual information: language-response length bin | 0.0448 |

## 7. Task Distribution

| Task | Count |
|---|---:|
| extractive_qa | 42 |
| bullet_points | 33 |
| summarize | 32 |
| short_answer_qa | 25 |
| mixed_language_qa | 24 |
| rewrite_patient_friendly | 23 |
| classify_doc_type | 16 |

## 8. Language Distribution

| Language | Count |
|---|---:|
| en | 114 |
| si | 44 |
| ta | 37 |
