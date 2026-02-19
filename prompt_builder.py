"""
prompt_builder.py
Builds grounded prompts for RAG.
"""
def build_prompt(query, retrieved_chunks):
    """
    query: user question (string)
    retrieved chunks: list of chunks from vector store
    """
    context_blocks = []

    for item in retrieved_chunks:
        chunk_text = item["chunk"]["text"]
        context_blocks.append(chunk_text)
    context = "\n\n".join(context_blocks)

    prompt = f"""
You are a helpful assistant.
Answer ONLY using the provided context.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question:
{query}

Answer:
"""

    return prompt.strip()