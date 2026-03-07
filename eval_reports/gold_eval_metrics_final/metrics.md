# Research Dataset Metrics

## 1. Dataset Cleaning Metrics

| Metric | Value |
|---|---:|
| Original total records | NA |
| Final total records | 99 |
| Removed records | NA |
| Retention rate | NA |
| Compression ratio | NA |
| Noise reduction rate | NA |
| Crawl-error records still present | 0 |

## 2. Refusal Metrics

| Metric | Value |
|---|---:|
| Total refusal-containing records | 15 |
| Answerable records (heuristic) | 30 |
| Over-refusal records | 0 |
| Over-Refusal Rate (ORR) | 0.0000 |

## 3. Diversity Metrics

| Metric | Value |
|---|---:|
| Normalized task entropy | 0.0000 |
| Linguistic Diversity Index (LDI) | 1.0000 |
| Source entropy (bits) | NA |
| Average assistant response length (words) | 6.3788 |
| Std assistant response length (words) | 5.0944 |
| Vocabulary size | 261 |
| Total tokens | 1538 |
| Type-Token Ratio (TTR) | 0.1697 |
| Vendi score (effective rank) | 23.2006 |
| Normalized Vendi score | 0.1172 |
| Vendi backend | sentence-transformers:all-MiniLM-L6-v2 |
| Vendi sample size | 198 |

## 4. Redundancy Metrics

| Metric | Value |
|---|---:|
| Duplicate response groups | 38 |
| Duplicated response items | 152 |
| Duplicate response rate | 0.7677 |
| Self-BLEU mean | 0.2755 |
| Self-BLEU std | 0.3308 |
| Self-BLEU samples used | 198 |

## 5. Grounding Metrics

| Metric | Value |
|---|---:|
| Mean context-answer similarity | 0.6009 |
| Std context-answer similarity | 0.2050 |
| Context-answer similarity backend | sentence-transformers:all-MiniLM-L6-v2 |
| Context-answer pairs used | 99 |
| Extractive QA records evaluated | 0 |
| Extractive exact substring match count | 0 |
| Extractive grounding accuracy | NA |

## 6. Dependency Metrics

| Metric | Value |
|---|---:|
| Mutual information: task-language | 0.0000 |
| Mutual information: task-response length bin | 0.0000 |
| Mutual information: language-response length bin | 0.0056 |

## 7. Task Distribution

| Task | Count |
|---|---:|
| short_answer_qa | 99 |

## 8. Language Distribution

| Language | Count |
|---|---:|
| en | 33 |
| si | 33 |
| ta | 33 |
