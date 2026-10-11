import tempfile

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging
from mock_data import practice_docs

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview", output_dimensionality=1536
)

llm = init_chat_model(
    model="gemini-3.5-flash", model_provider="google_genai", temperature=0.2
)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

chunks = text_splitter.split_documents(practice_docs)

vectore_store = Chroma.from_documents(
    documents=chunks, embedding=embedding_model, persist_directory=tempfile.mkdtemp()
)


def multi_query():
    retriever = MultiQueryRetriever.from_llm(
        retriever=vectore_store.as_retriever(search_kwargs={"k": 2}), llm=llm
    )


    docs = retriever.invoke("what is best programming language to start coding?")

    print(f"the length of docs {len(docs)}")

    for i, doc in enumerate(docs):
        print(
            f"{i + 1} metatdata: {doc.metadata.get('topic', 'N/A')} and page content: {doc.page_content}"
        )
        print("=" * 70)

