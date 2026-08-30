from dotenv import load_dotenv

load_dotenv()

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chat_models import init_chat_model


def get_tagline():
    """Get a tagline for the project."""
    prompt = ChatPromptTemplate.from_template("you are a very helpful assistant. give me just one tagline that can be use for this product name and target audience: {product_name}, {target} ")
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
    parser = StrOutputParser()

    # compose with pipe operator
    chain = prompt | model | parser

    inputs = [{"product_name": "dum biriyani", "target": "south indians"}, {"product_name": "ai assisstants", "target": "software devs"}]

    # invoke the chain with input
    result = chain.batch(inputs)

    for res in enumerate(result):
        print(f"{res[0]} ==> {res[1]}")
    

    return chain

# get_tagline()

# get tagline with new way of invoking models
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model


def get_tagline_new_way():
    # 1. Use .from_template() to create the prompt
    prompt = ChatPromptTemplate.from_template(
        "give only one tagline for this {product} targeting the audience {target}"
    )

    model = init_chat_model(
        model="gemini-2.5-flash", model_provider="google_genai", temperature=0.7, max_tokens=1500
    )
    parser = StrOutputParser()

    chain = prompt | model | parser

    inputs = [
        {"product": "dum biriyani", "target": "south indians"},
        {"product": "ai assisstants", "target": "software devs"},
    ]

    taglines = chain.batch(inputs)

    for i, tagline in enumerate(taglines):
        # 2. Fixed nested double-quote syntax error by using single quotes inside f-string
        print(f"{inputs[i]['product']} ==> {tagline}")

    return chain

get_tagline_new_way()