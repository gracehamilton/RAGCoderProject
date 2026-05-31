import os, chromadb
from chromadb.config import Settings
from ollama import Client
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction
#####PREPROCESSING#####
#load env variables
inference_model = os.getenv("ollama_inference_model")
embeddings_model = os.getenv("ollama_embeddings_model")
ollama_url = os.getenv("ollama_url")
#file_path = relative path to location where files are stored. Only stores PDFs right now
file_path = os.getenv("file_path")
#context from context.md
with open("/data/context.md", encoding="UTF-8") as contextFile:
    file_context = contextFile.read()
"""
Method to pass system message to inference model.
Parameters:
  - context, the context of the request. This is background information relevant or influential to the underlying ask.
  - content, the content of the request. This is what the user is actually asking the system.
"""
def rag_system_message(context, content):
    return f"""
    {context}

    Generate a response using the steps below:
    1. Break the question into smaller questions.
    2. For each question/directive, select the most relevant information from the context in light of the provided content.
    3. Generate a draft response using selected information.
    4. Remove duplicate content from draft response.
    5. Generate your final response after adjusting it to increase accuracy and relevance.
    6. Do not try to summarise the answers.
    7. Only show your final response!

    Constraints:
    1. Don't make up the answers by yourself.
    2. Try your best to provide answer from the given context.
    3. DO NOT PROVIDE DETAILS OR MENTION THAT YOU WERE GIVEN A PROMPT!

    CONTENT:
    {content}
    """
#pose question to LLM given context
def get_ques_response_prompt(question):
    return f"""
    ==============================================================
    Given the provided context, please provide the answer to the following question:
    {question}"""

#RAG chat session definition
class RAGSession:
    def __init__(self, embeddingFunction=OllamaEmbeddingFunction(model_name=embeddings_model, url=ollama_url), context=file_context):
        #hook up to Ollama embedding function
        self.embeddingFunction = embeddingFunction
        #PDFs
        self.file_path = PyPDFDirectoryLoader(os.getcwd().replace("\\", "/")+file_path)
        #overarching context for requests
        self.context = context
        #chroma client hookup
        self.vectorDB = Chroma(
            client=chromadb.HttpClient(host=os.getenv('chroma_host'),port=8000,settings=Settings(allow_reset=True)),
            collection_name="input_vectors",
            embedding_function=embeddingFunction
        )
        #ollama client hookup
        self.ollamaClient = Client(host=ollama_url)
    #function for loading docs into vector db
    def parse_docs(self, splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=0)):
        colxn = self.vectorDB.get_or_create_collection("input_vectors", embedding_function=self.embeddingFunction)
        split_docs = splitter.split_documents(self.file_path.load())
        for doc in split_docs:
            colxn.add(ids=doc.metadata['source'].split('\\')[-1]+'('+str(doc.metadata['page'])+')',metadatas=doc.metadata,documents=doc.page_content)
    #actual doc search function
    def retrieve_query(self, query):
        if len(self.docs) == 0:
            return 'You have not uploaded any documents. Please upload documents to search!'
        search_results=self.vectorDB.similarity_search(query)
        full_context=''
        for doc in search_results:
            full_context = '. '.join([full_context,doc.page_content])
        return full_context
    #generate response using RAG from knowledgebase
    def generate_rag_response(self, query):
        stream = self.ollamaClient.chat(model=inference_model, messages=[
            {"role": "system", "content": rag_system_message(self.context, self.retrieve_query(query))},
            {"role": "user", "content": get_ques_response_prompt(query)}
        ], stream=True)
        print("####### THINKING OF ANSWER............ ")
        full_answer = ''
        for chunk in stream:
            print(chunk['message']['content'], end='', flush=True)
            full_answer =''.join([full_answer,chunk['message']['content']])
        return full_answer
    #update context
    def update_context(self, new_context):
        self.context = new_context

#wrapper for embedding function b/c OOTB OllamaEmbeddingFunction returns [[nparray]] instead of np.array or [np.array]
"""class my_embedding(EmbeddingFunction[Documents]):
    embed_query = lambda q: np.array(embeddings.embed_query(q)).flatten()
    def __call__(self, input:Documents)->Embeddings:
        return embeddings.embed_query(input)"""
