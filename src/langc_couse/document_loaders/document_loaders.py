import os
import tempfile
from pathlib import Path

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
)
from langchain_core.documents import Document

load_dotenv()

def load_text_file():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        temp_file.write(b"this is a sample text file. \n This File is used to demonstrate the text loader capability")
        temp_file_path = temp_file.name

    try:
        loader = TextLoader(temp_file_path)
        documents = loader.load()

        for doc in documents:
            print("the doc content.....")
            print(doc)
            print("--------------------------")
            print(doc.page_content)

    finally:
        os.remove(temp_file_path)

# load_text_file()

def web_loader():
    loader = WebBaseLoader("https://en.wikipedia.org/wiki/Samara_Weaving", bs_kwargs={"parse_only": None})
    documents = loader.load()

    print(f"loaded {len(documents)} document(s) from web")
    print(f"source: {documents[0].metadata.get('source', 'N/A')}")
    print(f"contentlength: {len(documents[0].page_content)} characters")
    print(f"preview: {documents[0].page_content[:5000]}.....")

web_loader()

def lazy_loader():
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(5):
            path = Path(tmpdir)/ f"doc_{i}.txt"
            path.write_text(f"this is doc {i}. It contains sample content")

        loader = DirectoryLoader(tmpdir, glob="*.txt", loader_cls=TextLoader)

        for doc in loader.lazy_load():
            print("content preview for the doc ", doc.page_content[:100], " .....")
            print("metadata ", doc.metadata["source"])
            print(".....................................................")

# lazy_loader()

def doc_structure():
    doc = Document(
        page_content="This is a sample document",
        metadata={
            "source":"manual_text.txt",
            "author":"Paulo",
            "length":30,
            "tags":["sample", "document"],
            "created_at":"2026-09-10"
        }
    )

    print("the document structure")
    print("document content type ", type(doc.page_content))
    print(f"the page content {doc.page_content}")
    print("the metadata here is ", doc.metadata)

    updated_doc = Document(
        page_content=doc.page_content + " the updated document stuff",
        metadata={**doc.metadata, "updated": True}
    )

    print("the updated document....")
    print(f"the page content {updated_doc.page_content}")
    print("the metadata here is ", updated_doc.metadata)

# doc_structure()

def pdf_loader(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print(f"number of documents loaded: {len(documents)}")

    for i, doc in enumerate(documents):
        print(f"doc no: {i+1}, {doc.page_content[:1000]}")
        print(f"meta data is {doc.metadata}")
        print("------------------------------------------------------------")

# pdf_loader("./docs/langchain_demo.pdf")