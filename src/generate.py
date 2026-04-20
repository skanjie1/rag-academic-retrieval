"""
RAG generation pipeline using LangChain.
Augments LLM prompts with retrieved context from the FAISS index.
"""

import argparse
import yaml
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from retriever import FAISSRetriever


RAG_PROMPT_TEMPLATE = """You are a helpful research assistant specializing in NLP.
Answer the question based ONLY on the provided context from academic papers.
If the context doesn't contain enough information, say so clearly.

Context:
{context}

Question: {question}

Answer:"""


def build_rag_chain(config_path: str = "configs/config.yaml", use_finetuned: bool = False):
    """Build the RAG chain with retriever and LLM."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    retriever = FAISSRetriever(config_path, use_finetuned=use_finetuned)

    llm = ChatOpenAI(
        model=config["generation"]["llm_model"],
        temperature=config["generation"]["temperature"],
        max_tokens=config["generation"]["max_tokens"],
    )

    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    chain = (
        {
            "context": lambda x: retriever.retrieve_with_context(x["question"]),
            "question": lambda x: x["question"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


def query(question: str, use_finetuned: bool = False, verbose: bool = False) -> str:
    """Run a single query through the RAG pipeline."""
    chain, retriever = build_rag_chain(use_finetuned=use_finetuned)

    if verbose:
        results = retriever.retrieve(question)
        print(f"\n--- Retrieved {len(results)} chunks ---")
        for i, r in enumerate(results, 1):
            print(f"[{i}] score={r.score:.4f} | {r.metadata['title']}")
        print("---\n")

    answer = chain.invoke({"question": question})
    return answer


def main():
    parser = argparse.ArgumentParser(description="RAG Query Pipeline")
    parser.add_argument("--query", "-q", required=True, help="Question to answer")
    parser.add_argument("--finetuned", action="store_true", help="Use fine-tuned embeddings")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    answer = query(args.query, use_finetuned=args.finetuned, verbose=args.verbose)
    print(f"\nAnswer:\n{answer}")


if __name__ == "__main__":
    main()
