from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

import numpy as np

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview", output_dimensionality=768)

def batch_embeddings():
    text = [
        "what is machine learning?",
        "Shiva is a good boy",
        "How about a date? would you like to join me for dinner?"
    ]

    batch_embeddings = embeddings.embed_documents(text)

    for i, emb in enumerate(batch_embeddings):
        print(f"Text {i+1} - Vector dimensions: {len(emb)} ")
        print(f"Text {i+1} - First 5 values {emb[:5]}")
        print(f"Text {i+1} - Vector norm: {np.linalg.norm(emb):.4f} ")


batch_embeddings()