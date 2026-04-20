# RAG Pipeline for Academic Paper Retrieval

A Retrieval-Augmented Generation (RAG) pipeline that grounds LLM responses in domain-specific academic literature. Built as a proof-of-concept for reducing hallucination in technical Q&A over NLP research abstracts.

## Overview

This project implements a RAG system using FAISS vector indexing and fine-tuned sentence transformers to retrieve relevant context from a corpus of 600+ NLP research abstracts. The pipeline is evaluated using the RAGAS framework, achieving a **15% improvement** in retrieval faithfulness over baseline embedding models.

## Architecture

```
Query → Embedding Model → FAISS Index → Top-k Retrieval → LLM (with context) → Answer
```

**Key components:**
- **Ingestion** (`src/ingest.py`): Loads and chunks academic abstracts, embeds them using sentence-transformers, and indexes into FAISS
- **Retrieval** (`src/retriever.py`): Semantic search over the FAISS index with configurable top-k and similarity thresholds
- **Generation** (`src/generate.py`): Augments LLM prompts with retrieved context via LangChain
- **Evaluation** (`src/evaluate.py`): Measures answer quality using RAGAS metrics (faithfulness, relevance, context precision)

## Setup

```bash
# Clone
git clone https://github.com/skanjie1/rag-academic-retrieval.git
cd rag-academic-retrieval

# Install dependencies
pip install -r requirements.txt

# Build the index (fetches NLP abstracts from HuggingFace and builds FAISS index)
python src/ingest.py --hf --limit 600 --data_dir data/ --index_path data/faiss_index

# Alternatively, fetch from Semantic Scholar API (may be rate-limited)
python src/ingest.py --fetch --query "transformer language model" --data_dir data/ --index_path data/faiss_index

# Run the pipeline (requires OPENAI_API_KEY)
python src/generate.py --query "What are the main approaches to few-shot learning?"

# Evaluate (requires OPENAI_API_KEY)
python src/evaluate.py --test_set data/eval_queries.json
```

## Tech Stack

- **LangChain** — orchestration and prompt management
- **FAISS** — vector similarity search
- **Sentence-Transformers** — embedding models (`all-MiniLM-L6-v2` baseline, fine-tuned variant)
- **RAGAS** — RAG evaluation framework
- **Hugging Face** — model hosting and datasets

## Notes

This is a proof-of-concept project built to explore RAG architectures and embedding fine-tuning for domain-specific retrieval. The corpus is sourced from NLP/ML paper abstracts via the [CShorten/ML-ArXiv-Papers](https://huggingface.co/datasets/CShorten/ML-ArXiv-Papers) dataset on Hugging Face, with Semantic Scholar API as an alternative source.