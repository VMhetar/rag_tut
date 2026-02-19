"""
main.py
Runs the complete scratch RAG pipeline.
"""
from ingest import ingest_data, ingest_docx
from chunker import chunk_data
from vector_store import VectorStore
from embedder import embed_texts, normalize_embeddings
from prompt_builder import build_prompt
from llm_client import generate_response
import asyncio

def build_store(filepath):
    text = ingest_docx(filepath)

    documents = [{
        "id": "doc1",
        "text": text,
        "metadata": {}
    }]

    chunks = chunk_data(documents, chunk_size=1000, overlap=150)

    chunk_texts = [c["text"] for c in chunks]
    embeddings = embed_texts(chunk_texts)
    embeddings = normalize_embeddings(embeddings)

    store = VectorStore()
    store.add(embeddings, chunks)

    return store

def run_rag(filepath):
    store = build_store(filepath)

    print("Rag system ready. Type 'exit' to quit. \n")

    while True:
        query = input("Ask a question: ")

        if query.lower() == 'exit':
            break

        query_embedding = embed_texts(query)
        query_embedding = normalize_embeddings(query_embedding)

        results = store.search(query_embedding, top_k=3)

        prompt = build_prompt(query, results)

        answer = asyncio.run(generate_response(prompt))

        print("\nAnswer:\n")
        print(answer)
        print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    run_rag("data/ML notes.docx")