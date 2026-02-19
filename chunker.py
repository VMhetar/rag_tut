"""
Docstring for chunker
chunker.py is a module for chunking data into smaller units.
Chunking is helpful for processing large amounts of data in smaller chunks which can later given in RAG pipeline.
"""

def chunk_data(documents, chunk_size=500, overlap=50):  # Defines  the function chunk_data with parameters documents, chunk_size, and overlap
    chunks = [] # Creates an empty list

    for doc in documents:
        text = doc["text"] # Extracts the text from the documents
        doc_id = doc["id"] # Extracts the id from the documents

        start = 0 # Sets the start of the chunk
        chunk_index = 0 # Sets the index of the chunk

        while start < len(text):
            end = start + chunk_size # Creates the end of the chunk
            chunk_text = text[start:end] # Slices the text into chunks

            chunks.append({
                "id": f"{doc_id}_{chunk_index}", # Creates a unique id for each chunk Ex: if doc_id is 'notes' and chunk index is  then id is 'notes_0'
                "text": chunk_text, # Adds the chunk text
                "metadata": doc.get("metadata", {}) # Gets the metadata
            })
            start += chunk_size - overlap # Moves start forward with the overlap
            chunk_index += 1 # Increament in chunk index

    return chunks