import concurrent.futures

from langchain_community.document_loaders import WikipediaLoader

queries=["Samara Weaving", "Sydney Sweeney", "Lily James"]
all_docs=[]


def fetchWikiDocs(query: str):
    wikiLoader = WikipediaLoader(
        query=query, load_max_docs=2, doc_content_chars_max=1000
    )

    return wikiLoader.load()

with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    results = executor.map(fetchWikiDocs, queries)

    for docs in results:
        all_docs.extend(docs)

for doc in all_docs:
    print("***********************start of the section************")
    print(f"doc meta-data: {doc.metadata}")
    print("--------------------------------------------------------")
    print("--------------------------------------------------------")
    print(f"doc page content: {doc.page_content}")


