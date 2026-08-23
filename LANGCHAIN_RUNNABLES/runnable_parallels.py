from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

model = ChatGroq(model = 'openai/gpt-oss-20b')

parser = StrOutputParser()

tweet_prompt = PromptTemplate(
    template = 'Write the tweet on the following{topic}',
    input_variables = ['topic']
)


linkedin_prompt = PromptTemplate(
    template = 'Make a linkedin post on following {topic}',
    input_variables = ['topic']
)


tweet_chain = tweet_prompt | model | parser

linkedin_chain = linkedin_prompt | model | parser


parallel_chain = RunnableParallel(
    tweet = tweet_chain,
    linkedin = linkedin_chain
)


result = parallel_chain.invoke({'topic': 'AI'})
print('tweet: ')
print(result['tweet'])
print('linkedin: ')
print(result['linkedin'])