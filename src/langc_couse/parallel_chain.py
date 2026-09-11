from dotenv import load_dotenv

load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel


llm = init_chat_model(
        model="gemini-2.5-flash", model_provider="google_genai", temperature=0
    )
parser = StrOutputParser()

def demo_parallel_chain():

    summary_prompt = ChatPromptTemplate.from_template(
        "summarize the following text: {text}"
    )

    keywords_prompt = ChatPromptTemplate.from_template(
        "extract 5 keywords from the following text: {text}"
    )

    senitment_prompt = ChatPromptTemplate.from_template(
        "analyze the sentiment of the following text: {text}"
    )


    analysis_chain = RunnableParallel(
        summary_chain=summary_prompt | llm | parser,
        keywords_chain=keywords_prompt | llm | parser,
        sentiment_chain=senitment_prompt | llm | parser,
    )

    text = """
        The Rise of Remote Work and Its Long-Term Effects on Cities

        Over the past several years, remote work has shifted from a niche arrangement
        offered by a handful of tech companies to a mainstream expectation held by
        millions of employees worldwide. What began as a temporary necessity during
        a global health crisis has, for many organizations, become a permanent
        feature of how they operate. This shift has had ripple effects far beyond
        individual companies, reshaping commercial real estate, public transit
        systems, and even the demographic makeup of major cities.

        One of the most visible consequences has been the decline in demand for
        traditional office space. Many companies have downsized their physical
        footprints, opting for smaller offices or flexible co-working arrangements
        instead of long-term leases on entire floors of skyscrapers. This has left
        landlords in large metropolitan areas grappling with high vacancy rates,
        prompting some cities to explore converting empty office buildings into
        residential apartments, a process that is often more complicated and costly
        than it sounds due to differences in plumbing, window placement, and
        building codes between commercial and residential structures.

        Public transit systems have also felt the impact. Subway and bus systems
        that were designed around the assumption of a five-day commuting workweek
        have seen ridership patterns change dramatically, with many transit
        authorities now reporting their busiest days are Tuesday through Thursday,
        while Mondays and Fridays remain comparatively quiet. This has created
        budget challenges, since fare revenue in many cities funds a significant
        portion of transit operations, forcing some agencies to consider service
        cuts, fare increases, or seeking additional government subsidies to make
        up the shortfall.

        At the same time, remote work has enabled a wave of migration away from
        expensive coastal cities toward smaller towns and suburbs where housing is
        more affordable. This has brought new tax revenue and economic activity to
        previously overlooked areas, but it has also driven up housing costs in
        some of those same smaller communities, creating tension between long-time
        residents and newly arrived remote workers who often have higher salaries
        tied to jobs based in major cities.

        Not all effects have been negative, however. Employees who work remotely
        often report improved work-life balance, reduced commuting stress, and more
        flexibility to care for family members. Some studies have also found
        productivity gains in certain types of roles, particularly those involving
        independent, focus-heavy work, though collaborative and creative roles have
        sometimes suffered from reduced in-person interaction.

        As companies continue to experiment with hybrid models, fully remote
        policies, and mandatory return-to-office mandates, it remains unclear
        exactly what the long-term equilibrium will look like. What is clear is
        that the shift has permanently altered assumptions about where people live,
        how cities plan their infrastructure, and what employees expect from their
        employers.
    """

    results = analysis_chain.invoke({"text": text})

    print("*" * 80)
    print("--------------------------------- summary result start----------------------------------")
    print(results["summary_chain"])
    print("--------------------------------- summary result end----------------------------------")
    print("*" * 80)
    print("*" * 80)
    print("--------------------------------- keyword result start----------------------------------")
    print(results["keywords_chain"])
    print("--------------------------------- keyword result end----------------------------------")
    print("*" * 80)
    print("*" * 80)
    print("--------------------------------- sentiment result start----------------------------------")
    print(results["sentiment_chain"])
    print("--------------------------------- sentiment result end----------------------------------")
    print("*" * 80)

# demo_parallel_chain()

def demo_pass_though_chain():
    prompt = ChatPromptTemplate.from_template(
        "Original question: {question}\n"
        "context: {context}\n\n"
        "Answer the question based on the context"
    )

    def fake_retiever(input_dict):
        return "lang chain was created by Shiva Gaurav in 2022"

    chain = (
        RunnableParallel(
            context=RunnableLambda(fake_retiever), question=RunnablePassthrough()
        )
        | RunnableLambda(lambda x: {"context": x["context"], "question": x["question"]["question"]})
        | prompt | llm | parser
    )

    result = chain.invoke({"question": "who created langchain?"})
    print(f"Answer: {result}")

demo_pass_though_chain()
