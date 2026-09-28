import tempfile

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

load_dotenv()

# embedding model configuration
embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview", output_dimensionality=768
)

# llm
llm = init_chat_model(
    model="gemini-2.5-flash", model_provider="google_genai", temperature=0.2
)


# sample knowledge base
KNOWLEDGE_BASE = """# LangChain Framework

LangChain is a framework for developing applications powered by language models. It was created by Harrison Chase in October 2022.

## Core Components

1. **Models**: LangChain supports various LLM providers including OpenAI, Anthropic, and local models.

2. **Prompts**: Templates for structuring inputs to language models.

3. **Chains**: Sequences of calls to models and other components.

4. **Agents**: Systems that use LLMs to determine which actions to take.

5. **Memory**: Components for persisting state between chain/agent calls.

## LangGraph

LangGraph is a library for building stateful, multi-actor applications. Key features:
- State management
- Cycles and loops
- Human-in-the-loop
- Persistence

## Pricing

LangChain itself is open source and free. LangSmith (the observability platform) has a free tier and paid plans starting at $39/month.

## Getting Started

Install with: pip install langchain langchain-openai
Create your first chain in under 10 lines of code.
"""


# create a knowledge base
def create_knowledge_base():
    # knowledge docs
    doc = Document(
        page_content=KNOWLEDGE_BASE, metadata={"source": "langchain_knowledge_base.md"}
    )

    # split document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents([doc])

    # store in Chroma
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=tempfile.mkdtemp(),
    )

    # return vector store
    return vector_store


def demo_basic_rag():
    vector_store = create_knowledge_base()

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}, search_type="similarity"
    )

    # RAG prompt template

    prompt = ChatPromptTemplate.from_template(
        """
            Answer the question based only on the following context:

            {context}

            Question: {question}

            Answer:

            Make sure to answer in a concise manner, 
            and if you don't know the answer, just say "I don't know."""
    )

    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    # RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    #RAG TEST
    questions = [
        "What is LangChain?",
        "Who created LangChain?",
        "What is LangGraph used for?",
        "what is the capital of India",
    ]

    print("Basic RAG Demo:\n")
    for q in questions:
        answer = rag_chain.invoke(q)
        print(f"Q: {q}")
        print(f"A: {answer}\n")

# demo_basic_rag()



def demo_rag_with_resources():
    vector_store = create_knowledge_base()

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}, search_type="similarity"
    )

    # RAG with resources prompt template
    prompt = ChatPromptTemplate.from_template(
        """
            Answer the question based on the context below. Include which sources you used.

            Context:
            {context}

            Question: {question}

            Answer (include sources):"""
    )

    def format_with_sources(docs):
        formatted = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "unknown")
            formatted.append(f"[{i+1}] {source}:\n{doc.page_content}")
        return "\n\n".join(formatted)

    # RAG with resources chain
    rag_chain = (
        {"context": retriever | format_with_sources, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # RAG with resources TEST
    questions = [
        "What is LangChain?",
        "Who created LangChain?",
        "What is LangGraph used for?",
    ]

    print("RAG with Resources Demo:\n")
    for q in questions:
        answer = rag_chain.invoke(q)
        print(f"Q: {q}")
        print(f"A: {answer}\n")

# demo_rag_with_resources()


#structured RAG response

def demo_structured_rag():
    """RAG with structured output"""
    vector_store = create_knowledge_base()

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    class RAGOutput(BaseModel):
        answer: str = Field(description="The answer to the question")
        confidence: str = Field(description="High, medium or low")
        sources_used: list[str] = Field(description="The list of sources referenced")
        follow_up: str = Field(description="Suggested follow-up questions")

    structured_llm = llm.with_structured_output(RAGOutput)

    prompt = ChatPromptTemplate.from_template(
        """
            Answer the question based on the context below. Include which sources you used.
            Context: {context}
            Question: {question}
        Provide a structured response""")

    def format_docs(docs):
        formatted = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "unknown")
            formatted.append(f"[{i+1}] {source}:\n{doc.page_content}")
        return "\n\n".join(formatted)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | structured_llm
    )

    result = rag_chain.invoke("What is LangGraph?")

    print(f"answer:\n{result.answer}")
    print(f"confidence:\n{result.confidence}")
    print(f"sources_used:\n{result.sources_used}")
    print(f"follow_up:\n{result.follow_up}")

demo_structured_rag()
