# RAG Pipeline for Academic Paper Retrieval

A Retrieval-Augmented Generation (RAG) pipeline that grounds LLM responses in domain-specific academic literature. Built as a proof-of-concept for reducing hallucination in technical Q&A over NLP research abstracts.

## Overview

This project implements a RAG system using FAISS vector indexing and fine-tuned sentence transformers to retrieve relevant context from a corpus of 500+ NLP research abstracts. The pipeline is evaluated using the RAGAS framework, achieving a **15% improvement** in retrieval faithfulness over baseline embedding models.

## Architecture

```
Query → Embedding Model → FAISS Index → Top-k Retrieval → LLM (with context) → Answer
```

**Key components:**
- **Ingestion** (`src/ingest.py`): Loads and chunks academic abstracts, embeds them using sentence-transformers, and indexes into FAISS
- **Retrieval** (`src/retriever.py`): Semantic search over the FAISS index with configurable top-k and similarity thresholds
- **Generation** (`src/generate.py`): Augments LLM prompts with retrieved context via LangChain
- **Evaluation** (`src/evaluate.py`): Measures answer quality using RAGAS metrics (faithfulness, relevance, context precision)

## Results

| Metric | Baseline (`all-MiniLM-L6-v2`) | Fine-tuned |
|---|---|---|
| Context Precision | 0.72 | 0.83 |
| Faithfulness | 0.68 | 0.81 |
| Answer Relevancy | 0.74 | 0.82 |
| Context Recall | 0.65 | 0.77 |

Fine-tuning the embedding model on the academic corpus improved retrieval quality across all RAGAS metrics, with the largest gain in faithfulness (+13 points).

## Setup

```bash
# Clone
git clone https://github.com/skanjie1/rag-academic-retrieval.git
cd rag-academic-retrieval

# Install dependencies
pip install -r requirements.txt

# Download and prepare data
python src/ingest.py --data_dir data/ --index_path data/faiss_index

# Run the pipeline
python src/generate.py --query "What are the main approaches to few-shot learning?"

# Evaluate
python src/evaluate.py --test_set data/eval_queries.json
```

## Project Structure

```
├── configs/
│   └── config.yaml            # Model and retrieval hyperparameters
├── data/
│   └── README.md              # Data sourcing instructions
├── evaluation/
│   └── results.json           # RAGAS evaluation outputs
├── notebooks/
│   ├── 01_eda.ipynb           # Corpus exploration
│   ├── 02_embedding_analysis.ipynb  # Embedding space visualization
│   └── 03_evaluation.ipynb    # RAGAS evaluation analysis
├── src/
│   ├── ingest.py              # Data loading, chunking, indexing
│   ├── retriever.py           # FAISS retrieval logic
│   ├── generate.py            # LangChain RAG chain
│   ├── evaluate.py            # RAGAS evaluation pipeline
│   └── fine_tune.py           # Sentence transformer fine-tuning
├── requirements.txt
└── README.md
```

## Tech Stack

- **LangChain** — orchestration and prompt management
- **FAISS** — vector similarity search
- **Sentence-Transformers** — embedding models (`all-MiniLM-L6-v2` baseline, fine-tuned variant)
- **RAGAS** — RAG evaluation framework
- **Hugging Face** — model hosting and datasets

## Notes

This is a proof-of-concept project built to explore RAG architectures and embedding fine-tuning for domain-specific retrieval. The corpus is sourced from publicly available NLP paper abstracts via Semantic Scholar API.
