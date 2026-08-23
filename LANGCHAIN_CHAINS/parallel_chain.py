from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.runnables import RunnableParallel

load_dotenv()


# Model 1
llm1 = HuggingFaceEndpoint(
    repo_id='deepseek-ai/DeepSeek-V4-Flash-0731',
    task='text-generation'
)

model1 = ChatHuggingFace(llm=llm1)


# Model 2
llm2 = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    task='text-generation'
)

model2 = ChatHuggingFace(llm=llm2)


# Model 3 - Merge ke liye
model3 = ChatGroq(
    model='openai/gpt-oss-20b'
)


parser = StrOutputParser()


# Chain 1
prompt1 = PromptTemplate(
    template="Write a short summary about {topic}",
    input_variables=["topic"]
)

chain1 = prompt1 | model1 | parser


# Chain 2
prompt2 = PromptTemplate(
    template="List 3 important points about {topic}",
    input_variables=["topic"]
)

chain2 = prompt2 | model2 | parser


# Parallel
parallel_chain = RunnableParallel(
    summary=chain1,
    points=chain2
)


# Merge prompt
merge_prompt = PromptTemplate(
    template="""
    Create a final answer using the following two outputs.

    Summary:
    {summary}

    Important Points:
    {points}

    Combine them into one clear answer.
    """,
    input_variables=["summary", "points"]
)


# Merge chain
merge_chain = merge_prompt | model3 | parser


# Final chain
final_chain = parallel_chain | merge_chain


# Run
result = final_chain.invoke({
    "topic": "Convolutional Neural Network"
})

print(result)

final_chain.get_graph().print_ascii()