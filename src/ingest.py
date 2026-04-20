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


def fetch_abstracts(query: str, limit: int = 600) -> list[dict]:
    """Fetch paper abstracts from Semantic Scholar API."""
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    papers = []
    offset = 0
    batch_size = 100

    print(f"Fetching abstracts for query: '{query}'")
    while len(papers) < limit:
        params = {
            "query": query,
            "limit": min(batch_size, limit - len(papers)),
            "offset": offset,
            "fields": "title,abstract,authors,year,venue",
        }
        response = requests.get(url, params=params)
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

        if not data.get("data"):
            break

    print(f"Fetched {len(papers)} papers with valid abstracts")
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
    parser.add_argument("--query", default="natural language processing transformers")
    parser.add_argument("--limit", type=int, default=600)
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--index_path", default="data/faiss_index")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.fetch:
        papers = fetch_abstracts(args.query, args.limit)
        corpus_path = Path(args.data_dir) / "nlp_abstracts.json"
        corpus_path.parent.mkdir(parents=True, exist_ok=True)
        with open(corpus_path, "w") as f:
            json.dump(papers, f, indent=2)
    else:
        corpus_path = config["data"]["corpus_path"]
        with open(corpus_path, "r") as f:
            papers = json.load(f)

    chunks, metadatas = chunk_documents(papers, config)
    build_index(chunks, metadatas, config, args.index_path)


if __name__ == "__main__":
    main()
