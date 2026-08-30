from dotenv import load_dotenv

load_dotenv()

from importlib.metadata import version

from langchain_google_genai import ChatGoogleGenerativeAI

print(f"langchain-core version: {version("langchain-core")}")
print("LangGraph:", version("langgraph"))
print("LangChain Core:", version("langchain-core"))
print("LangChain Google Generative AI:", version("langchain-google-genai"))


def main() -> None:
    # llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    # response = llm.invoke("say 'setup complete' in one word")
    # print(f"OpenAI response: {response}")

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    response = llm.invoke("say 'setup complete' in one word")
    print(response.content)
    print("Setup complete!")


if __name__ == "__main__":
    main()
