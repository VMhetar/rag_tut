"""
Docstring for ingest
ingest.py is a module for ingesting data into the RAG pipeline.
"""

import os
from docx import Document # Necessary for word files.

def ingest_docx(filepath): # Function to ingest files
    document = Document(filepath) 
    full_text = []

    for paragraph in document.paragraphs: 
        full_text.append(paragraph.text) # Appends the text

    return "\n".join(full_text) # Returns the text


def ingest_data(directory_path):
    documents = [] # Creates an empty list for documents

    for filename in os.listdir(directory_path):
        if filename.endswith(".docx"):
            full_path = os.path.join(directory_path, filename) # Creates the full path
            text = ingest_docx(full_path)

            documents.append({
                "id": filename,
                "text": text,
                "metadata": {
                    "source": filename
                }
            })

    return documents
