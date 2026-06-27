from RAGService import RAGService
import os, re

#basic file reading/writing
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()
def write_file(file_path, content):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)
        file.close()

#RAG retrieval
def get_rag_response(rag_instance=None, query=None):
    if rag_instance is None:
        raise ValueError("RAG instance is not initialized.")
    if query is None:
        raise ValueError("Please provide a query.")
    try:
        response = rag_instance.generate_rag_response(query)
        return response
    except Exception as e:
        raise RuntimeError(f"Failed to get RAG response: {e}")
