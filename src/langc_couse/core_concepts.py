from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

def demo_basic_chain():
    """Demonstrate a basic chain with a prompt, model, and output parser."""
    # compoent 1: define the prompt template using LECL 
    prompt = ChatPromptTemplate.from_template("you are a very helpful assistant. now answer this question: {question} ")
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
    parser = StrOutputParser()

    # compose with pipe operator
    chain = prompt | model | parser

    #invoke the chain with input
    result = chain.invoke({"question": "What is the best way to transition to FDE role from a FE lead role"})
    print(f"result: {result}")

    return chain
# demo_basic_chain()

def demo_batch_execution():
    """Demonstrate batch execution of the chain with multiple inputs."""

    #compoent 1: define the prompt template using LECL
    prompt = ChatPromptTemplate.from_template("you are a very helpful assistant. Now give me just the translation of this text in Italian without any additional words, lines, etc.: {text} ")
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
    parser = StrOutputParser()

    # compose with pipe operator
    chain = prompt | model | parser

    inputs = [{"text": "what is your name?"}, {"text": "Hello, how are you?"}, {"text": "Good morning!"}, {"text": "join me for a dinner tonight."}]

    # invoke the chain with batch inputs
    results = chain.batch(inputs)

    for text in zip(inputs, results):
        print(f"{text[0]['text']} ==> {text[1]}")

    return chain
# demo_batch_execution()

def demo_streaming_execution():
    """""Demonstrate streaming execution of the chain with a single input."""

    #component 1: define the prompt template using LECL 
    prompt = ChatPromptTemplate.from_template("narrate me a funny story on this topic: {topic}")
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
    parser = StrOutputParser()

    # compose with pipe operator
    chain = prompt | model | parser

    # invoke the chain with streaming input
    stream = chain.stream({"topic": "a cat and a dog"})

    for chunk in stream:
        print(chunk, end="", flush=True)

    return chain

def demo_schema_inspection():
    """"Demonstrate schema inspection of the chain components."""""

    #component 1: define the prompt template using LECL
    prompt = ChatPromptTemplate.from_template("you are a very helpful assistant. now answer this question: {question} ")
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
    parser = StrOutputParser()

    # compose with pipe operator
    chain = prompt | model | parser

    # inspect the schema of the chain
    input_schema = chain.input_schema.model_json_schema()
    output_schema = chain.output_schema.model_json_schema()

    print(f"Input schema: {input_schema}")
    print(f"Output schema: {output_schema}")

    return chain

if __name__ == "__main__":
    demo_schema_inspection()


