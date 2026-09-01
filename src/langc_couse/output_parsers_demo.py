from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate


#json output parser
from langchain_core.output_parsers import JsonOutputParser

def demoJSONParser():
    prompt = ChatPromptTemplate.from_template(
        "give me a JSON object just with a 'name' and 'age' for description {description}"
    )

    model = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai", temperature=0)


    parser = JsonOutputParser()

    chain = prompt | model | parser

    result = chain.invoke({"description": "a person who is a software engineer and loves to code"})

    print(f"Result: {result}")

    return chain

demoJSONParser()