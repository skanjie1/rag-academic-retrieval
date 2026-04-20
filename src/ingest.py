"""
Data ingestion pipeline for the RAG system.
Fetches NLP paper abstracts, chunks them, generates embeddings, and builds a FAISS index.
"""

import json
import argparse
import requests
import numpy as np
import faiss
import yaml
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def fetch_abstracts_semantic_scholar(query: str, limit: int = 600) -> list[dict]:
    """Fetch paper abstracts from Semantic Scholar API."""
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    papers = []
    offset = 0
    batch_size = 100

    print(f"Fetching abstracts from Semantic Scholar for query: '{query}'")
    while len(papers) < limit:
        params = {
            "query": query,
            "limit": min(batch_size, limit - len(papers)),
            "offset": offset,
            "fields": "title,abstract,authors,year,venue",
        }
        try:
            response = requests.get(url, params=params, timeout=30)
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            break

        if response.status_code == 429:
            print("Rate limited by Semantic Scholar API.")
            break
        if response.status_code != 200:
            print(f"API error: {response.status_code}")
            break

        data = response.json()
        batch = [
            p for p in data.get("data", [])
            if p.get("abstract") and len(p["abstract"]) > 100
        ]
        papers.extend(batch)
        offset += batch_size
        print(f"  Fetched {len(papers)} papers so far...")

        if not data.get("data"):
            break

    print(f"Fetched {len(papers)} papers with valid abstracts")
    return papers[:limit]


def fetch_abstracts_huggingface(limit: int = 600) -> list[dict]:
    """Fetch NLP paper abstracts from Hugging Face datasets (fallback)."""
    from datasets import load_dataset

    print("Fetching abstracts from Hugging Face (CShorten/ML-ArXiv-Papers)...")
    dataset = load_dataset("CShorten/ML-ArXiv-Papers", split="train")

    # Filter for NLP/transformer-related papers
    nlp_keywords = [
        "language model", "transformer", "attention", "NLP", "natural language",
        "BERT", "GPT", "text", "token", "embedding", "sequence", "translation",
        "sentiment", "summarization", "question answering", "named entity",
    ]

    papers = []
    for item in dataset:
        title = item.get("title", "")
        abstract = item.get("abstract", "")
        if not abstract or len(abstract) < 100:
            continue

        # Check if paper is NLP-related
        combined = (title + " " + abstract).lower()
        if any(kw.lower() in combined for kw in nlp_keywords):
            papers.append({
                "title": title.strip(),
                "abstract": abstract.strip(),
                "year": None,
                "venue": "arXiv",
                "paperId": "",
            })

        if len(papers) >= limit:
            break

    print(f"Filtered {len(papers)} NLP-related papers from dataset")
    return papers[:limit]


def chunk_documents(papers: list[dict], config: dict) -> tuple[list[str], list[dict]]:
    """Split abstracts into chunks with metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunking"]["chunk_size"],
        chunk_overlap=config["chunking"]["chunk_overlap"],
        separators=["\n\n", "\n", ". ", " "],
    )

    chunks = []
    metadatas = []

    for paper in tqdm(papers, desc="Chunking"):
        text = f"Title: {paper['title']}\n\n{paper['abstract']}"
        paper_chunks = splitter.split_text(text)

        for i, chunk in enumerate(paper_chunks):
            chunks.append(chunk)
            metadatas.append({
                "title": paper["title"],
                "year": paper.get("year"),
                "venue": paper.get("venue", ""),
                "chunk_index": i,
                "paper_id": paper.get("paperId", ""),
            })

    return chunks, metadatas


def build_index(
    chunks: list[str],
    metadatas: list[dict],
    config: dict,
    output_dir: str,
) -> None:
    """Generate embeddings and build FAISS index."""
    if len(chunks) == 0:
        print("No chunks to index. Check that the fetch returned papers.")
        return

    model_name = config["embedding"]["base_model"]
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"Encoding {len(chunks)} chunks...")
    embeddings = model.encode(
        chunks,
        batch_size=config["embedding"]["batch_size"],
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = np.array(embeddings, dtype=np.float32)

    # Build FAISS index (inner product on normalized vectors = cosine similarity)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"FAISS index built with {index.ntotal} vectors (dim={dim})")

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(output_path / "index.faiss"))

    with open(output_path / "chunks.json", "w") as f:
        json.dump({"chunks": chunks, "metadatas": metadatas}, f, indent=2)

    print(f"Index and metadata saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="RAG Ingestion Pipeline")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--fetch", action="store_true", help="Fetch from Semantic Scholar API")
    parser.add_argument("--hf", action="store_true", help="Fetch from Hugging Face datasets (fallback)")
    parser.add_argument("--query", default="natural language processing transformers")
    parser.add_argument("--limit", type=int, default=600)
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--index_path", default="data/faiss_index")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.hf:
        papers = fetch_abstracts_huggingface(args.limit)
    elif args.fetch:
        papers = fetch_abstracts_semantic_scholar(args.query, args.limit)
        if not papers:
            print("Semantic Scholar failed. Falling back to Hugging Face dataset...")
            papers = fetch_abstracts_huggingface(args.limit)
    else:
        corpus_path = config["data"]["corpus_path"]
        with open(corpus_path, "r") as f:
            papers = json.load(f)

    if not papers:
        print("No papers fetched from any source.")
        return

    # Save corpus
    corpus_path = Path(args.data_dir) / "nlp_abstracts.json"
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    with open(corpus_path, "w") as f:
        json.dump(papers, f, indent=2)
    print(f"Saved {len(papers)} papers to {corpus_path}")

    chunks, metadatas = chunk_documents(papers, config)
    build_index(chunks, metadatas, config, args.index_path)


if __name__ == "__main__":
    main()