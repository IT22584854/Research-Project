SL\_Medical\_Corpus



A curated collection of Sri Lankan medical instruction datasets, preprocessing pipelines, and evaluation reports designed for training and evaluating document-grounded language models.



This repository contains the datasets, scripts, and analysis used to build and evaluate multilingual medical instruction datasets derived from Sri Lankan public health sources.



Project Overview



This project focuses on constructing high-quality instruction datasets grounded in Sri Lankan medical and public health documents.



The pipeline includes:



Document ingestion and cleaning



Instruction dataset generation



Multi-turn conversation construction



Refusal repair and regeneration



Dataset validation and quality checks



Evaluation and research metrics



The goal is to support document-grounded QA systems and instruction-tuned models for the healthcare domain.



Repository Structure

SL\_Medical\_Corpus

│

├── data/                 # Generated instruction datasets and curated data

│

├── scripts/              # Data processing and generation scripts

│

├── eval\_reports/         # Evaluation outputs and dataset metrics

│

├── requirements.txt      # Python dependencies

│

├── task\_distribution.png # Visualization of task distribution

│

└── validate\_repairs.py   # Dataset validation script

Dataset Description



The datasets consist of instruction-style records designed for training and evaluation of LLMs.



Each record follows a structured conversational format:



system – model instructions



user – task prompt and document context



assistant – grounded response



Example tasks include:



Document type classification



Extractive question answering



Short answer QA



Patient-friendly rewriting



Context-grounded follow-up questions



All responses are constrained to use only the provided document context.



Data Processing Pipeline



The dataset creation pipeline includes several stages:



Document collection and cleaning



Creation of the corpus



Instruction generation



Multi-turn dataset construction



Refusal detection and regeneration



Quality validation



Evaluation and metrics generation



Key evaluation metrics include:



Task distribution entropy



Duplicate response rate



Response length statistics



Refusal rate



Linguistic diversity



Installation



Clone the repository:



git clone https://github.com/IT22919700/SL\_Medical\_Corpus.git

cd SL\_Medical\_Corpus



Install dependencies:



pip install -r requirements.txt

Usage



Example script execution:



python scripts/research\_metrics.py \\

&nbsp; --in\_jsonl data/dataset.jsonl \\

&nbsp; --out\_dir eval\_reports/research\_metrics



Validation example:



python validate\_repairs.py

Notes



Large intermediate files and caches are excluded using .gitignore.



Sensitive configuration files such as .env are not included in the repository.



License



This repository is intended for research and academic use.



Author



Charunya Dissanayake

Data Science Research



Acknowledgements



This project builds on publicly available Sri Lankan medical and public health resources used to construct document-grounded datasets for research.

