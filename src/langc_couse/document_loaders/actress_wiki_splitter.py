import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import HTMLHeaderTextSplitter, RecursiveCharacterTextSplitter

urls = ["https://en.wikipedia.org/wiki/Samara_Weaving"]

headers_to_split_on = [("h2", "Section"), ("h3", "Sub Section")]

html_splitter = HTMLHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    return_each_element=False # false by default
)

char_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=50)

for url in urls:
    # 1. Fake a real web browser (Google Chrome on Windows)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers, timeout=10)

    # 1. Load the raw HTML into BeautifulSoup
    soup = BeautifulSoup(response.text, "html.parser")

    # 2. Target ONLY the main article content (ignores navbars and sidebars)
    # Wikipedia puts its main content in a div with id="mw-content-text"
    main_content = soup.find(id="mw-content-text")


    if main_content:
        # 3. Destroy unwanted tags to clean the text
        # - 'sup.reference': removes [1], [2] citation links
        # - 'span.mw-editsection': removes the [edit] buttons next to headers
        # - 'table': removes info boxes and tables that often format poorly as text

        for tag in main_content.select('sup.reference, span.mw-editsection, script, style, table, img, picture '):
            tag.extract() # This permanently deletes the tag from the soup tree

        # 4. Convert the cleaned soup back to an HTML string
        html_text = str(main_content)
        documents = html_splitter.split_text(html_text)

        final_chunks = char_splitter.split_documents(documents)

        for chunk in final_chunks:
            metadata = chunk.metadata
        
            section_name = metadata.get("Section", "")
        
            print("section name ------------> ", section_name)
            print("--------------------------------------------------------------------------")
            print(f"{chunk.page_content}...............")
            print(f"-----------------{section_name} section end ---------------------------------")
        

