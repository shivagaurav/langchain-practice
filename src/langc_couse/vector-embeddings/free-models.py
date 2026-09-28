from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text = "sample text to be embedded"

embedding = embeddings.embed_query(text)

print(f"the whole embedding in here.... {embedding}")