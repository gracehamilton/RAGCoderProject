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

tools = [
    {
        "name": "add_file_path",
        "description": "Adds a file path to the system.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The file path to add."
                }
            },
            "required": ["file_path"]
        }
    },
    {
        "name": "read_file",
        "description": "Reads the content of a file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The path to the file to read."
                }
            },
            "required": ["file_path"]
        }
    },
    {
        "name": "write_file",
        "description": "Writes content to a file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The path to the file to write."
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file."
                }
            },
            "required": ["file_path", "content"]
        }
    },
    {
        "name": "delete_file",
        "description": "Deletes a file from the system.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The path to the file to delete."
                }
            },
            "required": ["file_path"]
        }
    },
    {
        "name": "get_web_content",
        "description": "Fetches the content of a web page given its URL.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL of the web page to fetch."
                }
            },
            "required": ["url"]
        }
    },
    {
        "name": "parse_web_content",
        "description": "Parses HTML content and extracts text.",
        "input_schema": {
            "type": "object",
            "properties": {
                "html_content": {
                    "type": "string",
                    "description": "The HTML content to parse."
                }
            },
            "required": ["html_content"]
        }
    },
    {
        "name": "get_rag_response",
        "description": "Gets a response from the RAG module.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query to send to the RAG system."
                },
                "rag_instance": {
                    "type": "object",
                    "description": "The instance of the RAG system."
                }
            },
            "required": ["query", "rag_instance"]
        }
    }
]