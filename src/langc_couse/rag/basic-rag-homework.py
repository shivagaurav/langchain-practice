import tempfile

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

"""
EXERCISE: Build a complete document Q&A system that:
1. Takes a text document as input
2. Splits and embeds it
3. Allows multiple questions
4. Returns answers with confidence scores
"""

# create knowledge base
def create_kb(file_path: str, meta_source: str):
   #configure the embedding model
   embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview", output_dimensionality=1536)

   with open(file_path, 'r', encoding="utf-8") as f:
      text_content = f.read()

   #creating a document out of it
   doc = Document(page_content=text_content, metadata={"source": meta_source})

   #split the document into chunks
   text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
   chunks = text_splitter.split_documents([doc])

   #store the chunks in a vector knowledge base
   vector_store = Chroma.from_documents(
      documents=chunks,
      embedding=embedding_model,
      persist_directory=tempfile.mkdtemp()
   )

   # return vector store
   return vector_store


def chat_model():
   # configuring a chat model
   llm = init_chat_model(model="gemini-3.5-flash", model_provider="google_genai", temperature=0.2)

   #configuring prompt template
   prompt = ChatPromptTemplate.from_template(
      """
         Answer the question only based on the following context, do not use any other info apart from the context provided:
         {context}

         Question: {question}

         You can be creative if it fits the context,
         but if you don't know the answer explicitly state that you don't know the answer 
      """
   )

   # creating knowledge retriever
   vector_store = create_kb(file_path="./sydney_sweeney_profile.txt", meta_source="sydney sweeney profile")
   retriever = vector_store.as_retriever(search_kwargs={"k": 5}, search_type="similarity")

   #format vector store embeddings
   def format_docs(docs):
      return "\n\n".join(doc.page_content for doc in docs)

   rag_chain = (
      {"context": retriever | format_docs, "question": RunnablePassthrough()}
      | prompt
      | llm
      | StrOutputParser()
   )

   questions = [
      "what is sydney sweeney known for ?",
      "who are sweeney's main competitors?",
      "Where did sydney get her first break through?",
      "List some of the career highlights in sydney's life",
      "Since when did Sydney start getting huge attention"
   ]

   for q in questions:
      answer = rag_chain.invoke(q)
      print("________________________________________________________________")
      print(f"Q: {q}\n")
      print(f"A: {answer}\n")

chat_model()