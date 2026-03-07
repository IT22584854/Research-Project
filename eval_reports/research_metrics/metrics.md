# Research Dataset Metrics

## 1. Dataset Cleaning Metrics

| Metric | Value |
|---|---:|
| Original total records | 4887 |
| Final total records | 3894 |
| Removed records | 6 |
| Retention rate | 0.7968 |
| Compression ratio | 0.7968 |
| Noise reduction rate | 0.0012 |
| Crawl-error records still present | 0 |

## 2. Refusal Metrics

| Metric | Value |
|---|---:|
| Total refusal-containing records | 499 |
| Answerable records (heuristic) | 2888 |
| Over-refusal records | 0 |
| Over-Refusal Rate (ORR) | 0.0000 |

## 3. Diversity Metrics

| Metric | Value |
|---|---:|
| Normalized task entropy | 0.9762 |
| Linguistic Diversity Index (LDI) | 0.9683 |
| Source entropy (bits) | NA |
| Average assistant response length (words) | 36.9452 |
| Std assistant response length (words) | 50.3624 |
| Vocabulary size | 12472 |
| Total tokens | 453409 |
| Type-Token Ratio (TTR) | 0.0275 |
| Vendi score (effective rank) | 55.6355 |
| Normalized Vendi score | 0.1113 |
| Vendi backend | sentence-transformers:all-MiniLM-L6-v2 |
| Vendi sample size | 500 |

## 4. Redundancy Metrics

| Metric | Value |
|---|---:|
| Duplicate response groups | 623 |
| Duplicated response items | 2622 |
| Duplicate response rate | 0.3367 |
| Self-BLEU mean | 0.0866 |
| Self-BLEU std | 0.1246 |
| Self-BLEU samples used | 300 |

## 5. Grounding Metrics

| Metric | Value |
|---|---:|
| Mean context-answer similarity | 0.4383 |
| Std context-answer similarity | 0.2728 |
| Context-answer similarity backend | sentence-transformers:all-MiniLM-L6-v2 |
| Context-answer pairs used | 1000 |
| Extractive QA records evaluated | 648 |
| Extractive exact substring match count | 30 |
| Extractive grounding accuracy | 0.0463 |

## 6. Dependency Metrics

| Metric | Value |
|---|---:|
| Mutual information: task-language | 0.0022 |
| Mutual information: task-response length bin | 0.2392 |
| Mutual information: language-response length bin | 0.0254 |

## 7. Task Distribution

| Task | Count |
|---|---:|
| short_answer_qa | 726 |
| rewrite_patient_friendly | 662 |
| extractive_qa | 648 |
| summarize | 612 |
| bullet_points | 548 |
| mixed_language_qa | 477 |
| classify_doc_type | 221 |

## 8. Language Distribution

| Language | Count |
|---|---:|
| en | 1792 |
| ta | 1086 |
| si | 1016 |
