from RAGService import RAGService
from bs4 import BeautifulSoup as bs
import os, requests

#basic file access/creation/reading/writing
def add_file_path(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file path '{file_path}' does not exist.")
    os.path.join(file_path)
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()
def write_file(file_path, content):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)
        file.close()
def delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
    else:
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")

#scraping + parsing web content
def get_web_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses
        return response.text
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to fetch content from {url}: {e}")
def parse_web_content(html_content):
    soup = bs(html_content, 'html.parser')
    return soup.get_text(separator=' ', strip=True)

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
