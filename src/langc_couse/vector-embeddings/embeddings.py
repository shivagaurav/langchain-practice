from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview", output_dimensionality=768)

text = "sample text to be embedded"

embedding = embeddings.embed_query(text)
print(f"embedding for single text: {len(embedding)}")

