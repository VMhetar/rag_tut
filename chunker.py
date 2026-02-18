"""
Docstring for chunker
chunker.py is a module for chunking data into smaller units.
Chunking is helpful for processing large amounts of data in smaller chunks which can later given in RAG pipeline.
"""

def chunk_data(documents, chunk_size=500, overlap=50):
    chunks = []

    for doc in documents:
        text = doc["text"]
        doc_id = doc["id"]

        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            chunks.append({
                "id": f"{doc_id}_{chunk_index}",
                "text": chunk_text,
                "metadata": doc.get("metadata", {})
            })
            start += chunk_size - overlap
            chunk_index += 1

    return chunks