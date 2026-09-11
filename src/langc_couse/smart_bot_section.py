""" "
this will be a smart bot implementation
The idea is to have a smart bot which is capable of understanding and
responding to user queries intelligently.
"""

import os
from typing import Optional, List

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith import traceable, Client
from pydantic import BaseModel, Field

load_dotenv()

# --- Langsmith configuration ---
if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
    os.environ.setdefault("LANGSMITH_PROJECT_NAME", "Smart Q&A bot project")
    print(
        f"Langsmith configuration set. Project name --- {os.environ.get('LANGSMITH_PROJECT_NAME')}"
    )


# schema definition
class QAResponse(BaseModel):
    answer: str = Field(description="The answer to the user's question")
    confidence: str = Field(description="Confidence level: high, medium or low")
    reasoning: str = Field(description="The reasoning behind the answer provided")
    follow_up_questions: list[str] = Field(
        description="List of follow-up questions related to the user's query",
        default_factory=list,
    )
    sources_needed: bool = Field(
        description="Indicates whether additional sources are needed for the answer",
        default=False,
    )


# Bot implementation
class SmartQABot:
    def __init__(self, model_name: str = "gemini-3.5-flash", temperature: float = 0.7):
        self.model = ChatGoogleGenerativeAI(
            model=model_name, temperature=temperature
        ).with_structured_output(QAResponse)

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                        You are a knowledgeable Q&A assisstant

                        Your guidelines:
                        - Provide accurate and concise answers.
                        - If unsure, indicate the uncertainty rather than providing potentially incorrect information.
                        - Always provide reasoning for your answers.
                        - Suggest follow-up questions when appropriate.
                        - Indicate when additional sources are needed for the answer.

                        Always respond with accurate, helpful information
                    """,
                ),
                ("human", "{question}"),
            ]
        )

        self.chain = self.prompt | self.model

    @traceable(name="ask question", run_type="chain")
    def ask(self, question: str) -> QAResponse:
        try:
            response = self.chain.invoke({"question": question})
            return response
        except Exception as e:
            print(f"An error occurred: {e}")
            return QAResponse(
                answer="I'm sorry, couldn't process your question at this time",
                confidence="low",
                reasoning=str(e),
                follow_up_questions=[],
                sources_needed=True,
            )

    @traceable(name="ask batch questions", run_type="chain")
    def ask_batch(self, questions: list[str]) -> list[QAResponse]:
        """ " ask multiple questions in parallel"""
        try:
            inputs = [{"question": q} for q in questions]
            return self.chain.batch(inputs)
        except Exception as e:
            print(f"An error occurred: {e}")
            return [
                QAResponse(
                    answer="I'm sorry, couldn't process your question at this time",
                    confidence="low",
                    reasoning=str(e),
                    follow_up_questions=[],
                    sources_needed=True,
                )
            ]

    def ask_stream(self, question: str):
        """ask a question and stream the response"""
        try:
            for chunk in self.chain.stream({"question": question}):
                yield chunk
        except Exception as e:
            print(f"An error occurred: {e}")
            yield QAResponse(
                answer="I'm sorry, couldn't process your question at this time",
                confidence="low",
                reasoning=str(e),
                follow_up_questions=[],
                sources_needed=True,
            )


def demo_qa_bot():
    bot = SmartQABot()
    questions = [
        "what is best ice-cream hub in Bengaluru - I want the oldest and most famous one",
        "Explain the theory of relativity",
        "What is the capital of France?",
    ]

    for question in questions:
        print(f"Processing question: {question}")
        print("-" * 60)
        print("-" * 60)
        response = bot.ask(question)
        print("-" * 60)
        print(f"Question: {question}")
        print("-" * 40)
        print(f"Answer: {response.answer}")
        print("-" * 40)
        print(f"Confidence: {response.confidence}")
        print("-" * 40)
        print(f"Reasoning: {response.reasoning}")
        print("-" * 40)
        print(f"Follow-up Questions: {response.follow_up_questions}")
        print("-" * 40)
        print(f"Sources Needed: {response.sources_needed}")
        print("-" * 60)
        print("-" * 60)


def demo_qa_bot_batch():
    bot = SmartQABot()

    questions = [
        "what is best ice-cream hub in Bengaluru - I want the oldest and most famous one",
        "Explain the theory of relativity",
        "What is the capital of France?",
    ]

    print(f"Processing batch questions: {questions}")
    print("-" * 60)
    responses = bot.ask_batch(questions)
    for question, response in zip(questions, responses):
        print("*" * 60)
        print("*" * 60)
        print(f"Question: {question}")
        print("-" * 40)
        print(f"Answer: {response.answer}")
        print("-" * 40)
        print(f"Confidence: {response.confidence}")
        print("-" * 40)
        print(f"Reasoning: {response.reasoning}")
        print("-" * 40)
        print(f"Follow-up Questions: {response.follow_up_questions}")
        print("-" * 40)
        print(f"Sources Needed: {response.sources_needed}")
        print("*" * 60)
        print("*" * 60)


def demo_qa_bot_stream():
    bot = SmartQABot()
    question = "what is the latest movie of lokesh kanagaraj and what is the story about?"
    print(f"Processing question: {question}")
    print("-" * 60)
    latest_chunk = None
    for i, chunk in enumerate(bot.ask_stream(question)):
        print(f"Chunk[{i}] of the chunk: {chunk}")
        latest_chunk = chunk
        print("*" * 40)

    print("-" * 60)
    print("-" * 60)
    print("Final response:")
    print("*" * 80)
    print(f"answer: {latest_chunk.answer}")
    print("*" * 80)
    print(f"confidence: {latest_chunk.confidence}")
    print("*" * 80)
    print(f"reasoning: {latest_chunk.reasoning}")
    print("*" * 80)
    print(f"follow_up_questions: {latest_chunk.follow_up_questions}")
    print("*" * 80)
    print(f"sources_needed: {latest_chunk.sources_needed}")
    print("*" * 80)
    print("-" * 60)

if __name__ == "__main__":
    try:
        # demo_qa_bot()
        # demo_qa_bot_batch()
        demo_qa_bot_stream()
    finally:
        Client().flush()
