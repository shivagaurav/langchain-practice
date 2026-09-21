from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("./docs/imf_report.pdf")
pages = loader.load()

print(f"total number pages loaded: {len(pages)}")

# Test how it handled the first content page
print(pages[61].page_content)