"""
FAISS-based semantic retriever for the RAG pipeline.
Supports both baseline and fine-tuned embedding models.
"""

import json
import numpy as np
import faiss
import yaml
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer


@dataclass
class RetrievalResult:
    chunk: str
    metadata: dict
    score: float


class FAISSRetriever:
    def __init__(self, config_path: str = "configs/config.yaml", use_finetuned: bool = False):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        model_key = "fine_tuned_model" if use_finetuned else "base_model"
        model_name = self.config["embedding"][model_key]
        self.model = SentenceTransformer(model_name)

        index_dir = self.config["data"]["index_path"]
        self.index = faiss.read_index(f"{index_dir}/index.faiss")

        with open(f"{index_dir}/chunks.json", "r") as f:
            data = json.load(f)
            self.chunks = data["chunks"]
            self.metadatas = data["metadatas"]

    def retrieve(self, query: str, top_k: int = None) -> list[RetrievalResult]:
        """Retrieve top-k most relevant chunks for a query."""
        top_k = top_k or self.config["retrieval"]["top_k"]
        threshold = self.config["retrieval"]["similarity_threshold"]

        query_embedding = self.model.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or score < threshold:
                continue
            results.append(RetrievalResult(
                chunk=self.chunks[idx],
                metadata=self.metadatas[idx],
                score=float(score),
            ))

        return results

    def retrieve_with_context(self, query: str, top_k: int = None) -> str:
        """Retrieve and format context string for LLM prompt augmentation."""
        results = self.retrieve(query, top_k)

        if not results:
            return "No relevant context found."

        context_parts = []
        for i, r in enumerate(results, 1):
            source = r.metadata.get("title", "Unknown")
            year = r.metadata.get("year", "")
            context_parts.append(
                f"[{i}] (Source: {source}, {year})\n{r.chunk}"
            )

        return "\n\n".join(context_parts)
