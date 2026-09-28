from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


"""
    EXERCISE: Create a complete vector store setup that:
    1. Takes a list of text strings
    2. Splits them into chunks
    3. Stores in Chroma
    4. Returns a configured retriever

    Test with sample documents.
"""

embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview", output_dimensionality=768)

def create_retriever(texts: list[str], chunk_size: int = 1000, chunk_overlap: int = 200):
    #convert text to document objects
    documents = [Document(page_content=text) for text in texts]

    #split document to chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(documents)

    #store in Chroma
    vector_store = Chroma.from_documents(chunks, )

    #return retriever
    return vector_store.as_retriever(search_kwargs={"k": 3}, search_type="similarity")

sample_texts = [
        "Python is a versatile programming language used in web development, "
        "data science, machine learning, and automation. It has a simple syntax "
        "that makes it easy to learn and read.",
        "JavaScript is the language of the web. It runs in browsers and on "
        "servers with Node.js. Modern frameworks like React and Vue make "
        "building web applications efficient.",
        "Rust is a systems programming language focused on safety and "
        "performance. It prevents common bugs like null pointer dereferences "
        "and data races at compile time.",
]