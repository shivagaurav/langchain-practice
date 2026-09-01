from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chat_models import init_chat_model

load_dotenv()


def prompt_messages():
    prompt = ChatPromptTemplate.from_template(
        "tell me a {adjective} joke about {topic}"
    )
    message = prompt.format_messages(adjective="funny", topic="programming")

    print(f"Prompt messages: {message}")

    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
    parser = StrOutputParser()

    result = model.invoke(message)
    parsed_result = parser.parse(result.content)

    print(f"Parsed result: {parsed_result}")


# prompt_messages()

"""
gng to demo the FewShotChatMessagePromptTemplate now, where the llm will learn how to respond using
the examples provided in the prompt. The llm will then be able to respond to new prompts in a similar manner. 
"""


def few_shot_prompt_messages():
    examples = [
        {"input": "happy", "output": "sad"},
        {"input": "good", "output": "bad"},
        {"input": "fast", "output": "slow"},
    ]

    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}"),
    ])

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )

    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "give me the opposite of each word"),
            few_shot_prompt,
            ("human", "{input}")
        ]
    )

    model = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai" , temperature=0.7)
    response = model.invoke(final_prompt.format_messages(input="happy"))

    print(f"Few-shot prompt response: {response.content}")

    few_shot_prompt_messages()