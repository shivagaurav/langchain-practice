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

# demoJSONParser()

""""demo the use of a pydantic parser"""
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


def usePydanticParser():

    model = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai", temperature=0)
    class Person(BaseModel):
        name: str = Field(description="The name of the person")
        age: int = Field(description="The age of the person")
        occupation: str = Field(description="The occupation of the person")

    parser = PydanticOutputParser(pydantic_object=Person)

    prompt = ChatPromptTemplate.from_template(
        "give me a JSON object just with a 'name', 'age', and 'occupation' for description {description}"
    )

    chain = prompt | model | parser

    result = chain.invoke({"description": "a 30 year old artist named Maria"})

    print(f"Result: {result}")

    return chain

# usePydanticParser()

def structuredParserDemo():
    llm = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai", temperature=0.7)

    class MovieReview(BaseModel):
        title: str = Field(description="The title of the movie")
        review: str = Field(description="a creative and a bit detailed summary of the movie review")
        rating: int = Field(description="The rating of the movie on a scale of 1 to 10")

    structured_model = llm.with_structured_output(MovieReview)

    result = structured_model.invoke("review: Inception is a mind-bending thriller, not for the weak hearted 9/10")
    print(f"Result: {result}")

structuredParserDemo()