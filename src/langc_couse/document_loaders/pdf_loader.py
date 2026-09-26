from langchain_community.document_loaders import PyPDFLoader
from langchain_unstructured import UnstructuredLoader


def pdfLoader(): 
    loader = PyPDFLoader("./docs/imf_report.pdf")
    pages = loader.load()

    print(f"total number pages loaded: {len(pages)}")

    # Test how it handled the first content page
    print(pages[61].page_content)
# pdfLoader()

def unstructuredPdfLoader():
    loader = UnstructuredLoader("./docs/imf_report.pdf", strategy="fast")
    pages = loader.load()

    print(f"total number pages loaded: {len(pages)}")

    # Test how it handled the first content page
    # for doc in pages[:10]:
    #     category = doc.metadata.get("category")
    #     print(f"[{category}] {doc.page_content[:50]}...")
unstructuredPdfLoader()