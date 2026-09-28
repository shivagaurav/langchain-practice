import os
import tempfile

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview", output_dimensionality=768)

SAMPLE_DOCS = [
    Document(
        page_content="LangChain is a framework for developing applications powered by language models.",
        metadata={"source": "langchain_docs", "topic": "overview"},
    ),
    Document(
        page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs.",
        metadata={"source": "langgraph_docs", "topic": "overview"},
    ),
    Document(
        page_content="Vector stores are databases optimized for storing and searching embeddings.",
        metadata={"source": "vector_guide", "topic": "database"},
    ),
    Document(
        page_content="RAG combines retrieval with generation for more accurate LLM responses.",
        metadata={"source": "rag_guide", "topic": "architecture"},
    ),
    Document(
        page_content="Embeddings convert text into numerical vectors for semantic similarity.",
        metadata={"source": "embeddings_guide", "topic": "fundamentals"},
    ),
    Document(
        page_content="Chroma is an open-source embedding database for AI applications.",
        metadata={"source": "chroma_docs", "topic": "database"},
    ),
    Document(
        page_content="FAISS is a library for efficient similarity search developed by Facebook.",
        metadata={"source": "faiss_docs", "topic": "database"},
    ),
    Document(
        page_content="Pinecone is a managed vector database service for production workloads.",
        metadata={"source": "pinecone_docs", "topic": "database"},
    ),
]


def chroma_basics():
    with tempfile.TemporaryDirectory() as tmpdir:
        #create vector store from documents
        vectorStore = Chroma.from_documents(embedding=embedding_model, documents=SAMPLE_DOCS, persist_directory=tmpdir)
       
        print(f"vector store created and {vectorStore._collection.count()} persisted")

        #perform similarity search
        query = "what is langchain?"
        results = vectorStore.similarity_search(query, k=2)

        print(f"top two results of the query: {query}")

        for i, doc in enumerate(results):
            print(f"result {i+1} content: {doc.page_content} and metadata source: {doc.metadata['source']}")


# chroma_basics()

def similarity_search():
    with tempfile.TemporaryDirectory() as tmpdir:
        #create vector store from documents
        vectorStore = Chroma.from_documents(embedding=embedding_model, documents=SAMPLE_DOCS, persist_directory=tmpdir)

        query = "Explain vector stores"

        results = vectorStore.similarity_search_with_score(query, k=3)

        print(f"top 3 results for the query: {query}")

        for i, (doc, score) in enumerate(results):
            print(f"result {i+1} socre: {score:.4f} content: {doc.page_content} metadata: {doc.metadata['source']}")

# similarity_search()

def metadata_filtering():
    with tempfile.TemporaryDirectory() as tmpdir:
        vectorStore = Chroma.from_documents(embedding=embedding_model, documents=SAMPLE_DOCS, persist_directory=tmpdir)

        query = "what are the differnt types of databases available"

        simpleSearchResults = vectorStore.similarity_search(query, k=5)

        print(f"top 5 results of the query: {query}")
        
        for i, doc in enumerate(simpleSearchResults):
            print(f"result {i+1} content: {doc.page_content} and metadata source: {doc.metadata['source']}")

        filterCriteria = {"topic": "database"}

        criteriaSearch = vectorStore.similarity_search(query, k=5, filter=filterCriteria)

        print(f"top 5 results of the filter search query: {query}")
                
        for i, doc in enumerate(criteriaSearch):
            print(f"result {i+1} content: {doc.page_content} and metadata source: {doc.metadata['source']}")

# metadata_filtering()

def persist_chroma():
    persist_dir = "./chroma_db/"

    vectorStore = Chroma.from_documents(embedding=embedding_model, documents=SAMPLE_DOCS, persist_directory=persist_dir)

    print(f"the persisted document count is: {vectorStore._collection.count()}")

    # simulate restart

    reloaded = Chroma(embedding_function=embedding_model, persist_directory=persist_dir)

    query="What is a RAG"

    results = reloaded.similarity_search(query, k=2)

    for i, doc in enumerate(results):
        print(f"result {i+1}: content is {doc.page_content}")

persist_chroma()


