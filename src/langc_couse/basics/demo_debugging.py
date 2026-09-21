from dotenv import load_dotenv

load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

model = init_chat_model(
    model="gemini-3.5-flash", model_provider="google_genai", temperature=0
)

parser = StrOutputParser()


def demo_debugging():
    prompt = ChatPromptTemplate.from_template("say hello to {name}")
    chain = prompt | model | parser


    # get the schema
    print("chain input schema: ", chain.input_schema.model_json_schema())
    print("chain output schema ", chain.output_schema.model_json_schema())

    result = chain.with_config(run_name="greeting_chain").invoke({"name": "Alice"})

    print(f"greeting: {result}")

    def log_step(x, step_name=""):
        print(f"[{step_name}] {type(x).__name__}: {str(x)[:100]}")
        return x

    debug_chain = (
        prompt
        | RunnableLambda(lambda x: log_step(x, "after prompt"))
        | model
        | RunnableLambda(lambda x: log_step(x, "after model"))
        | parser
    )

    debug_chain.invoke({"name": "debug"})


if __name__ == "__main__":
    demo_debugging()
