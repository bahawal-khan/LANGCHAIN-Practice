from langchain_groq import ChatGroq
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import (
    RunnableSequence,
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
)

load_dotenv()

model = ChatGroq(model = 'openai/gpt-oss-20b')

parser = StrOutputParser()


# -------------------------
# 1. Prompt
# -------------------------


article_prompt = PromptTemplate(
    template = 'write a short article about {topic}',
    input_variables = ['topic']
)


article_chain = (
    article_prompt | model | parser
)

# -------------------------
# 3. Parallel Processing
# -------------------------


summary_prompt = PromptTemplate(
    template = "Summarize this article:\n{text}",
    input_variables = ['text']
)


keyword_prompt =  PromptTemplate(
    template = "Give 5 important keywords from this article:\n{text}",
    input_variables =  ['text']
)


summary_chain = summary_prompt | model | parser

keyword_chain = keyword_prompt | model | parser



parallel_chain = RunnableParallel(
    orignal = RunnablePassthrough(),
    summary = summary_chain,
    keyword = keyword_chain
)


# -------------------------
# 4. Lambda
# -------------------------


def clean_result(result):
    return{
        'article': result['orignal'],
        'summary': result['summary'],
        'keyword': result['keyword']
    }

lambda_chain = RunnableLambda(clean_result)


#-------------------------
# 5. Complete Sequence
# -------------------------


final_chain = RunnableSequence(
         article_chain,
         parallel_chain,
         lambda_chain
)


result = final_chain.invoke({'topic': 'Convolution neural network'})
print('article: ')
print(result['article'])
print('summary: ')
print(result['summary'])
print('keyword: ')
print(result['keyword'])

