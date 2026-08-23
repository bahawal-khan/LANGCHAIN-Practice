from langchain_groq import ChatGroq
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch,RunnableLambda,RunnableSequence

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-20b")

parser = StrOutputParser()



# Prompt 1: Content generate karo
content_prompt = PromptTemplate(
    template="Write an article about {topic}",
    input_variables=["topic"]
)


content = content_prompt | model | parser


summary_prompt = PromptTemplate(
    template="Summarize the following article:\n{text}",
    input_variables=["text"]
)


summary = summary_prompt | model | parser

check_length = RunnableBranch((
    lambda text: len(text.split())>500,
    RunnableLambda(lambda text: {'text': text}) | summary
),

RunnableLambda(lambda text: text)

)


chain = RunnableSequence(
    content,
    check_length
)


result = chain.invoke({'topic': 'cricket'})
print(result)