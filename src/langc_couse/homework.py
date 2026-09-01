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

# get_tagline_new_way()


def excerise_multi_models():
    """
    EXERCISE: Create a function that:
         1. Takes a question and a list of model names
         2. Gets responses from all models
         3. Returns a dict of {model_name: response}
     
         Test with: question="What is AI?", models=["gpt-4o-mini", "gpt-4o"]
    """
    def get_responses(qtn: str, model_names: list[str]) -> dict[str, str]:
        responses = {}
        for model_name in model_names:
            model = init_chat_model(
                model=model_name,
                model_provider="google_genai",
                temperature=0.7,
                streaming=False,
            )
            response = model.invoke(qtn)
            responses[model_name] = response.content
        return responses

    # Test the function
    results = get_responses("What is AI?", ["gemini-2.5-flash", "gemini-3.5-flash"])
    for model, answer in results.items():
        print(f"Response from {model}: {answer}\n")

excerise_multi_models()