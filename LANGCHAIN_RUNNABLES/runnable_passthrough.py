from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-20b")


parser = StrOutputParser()


summary_prompt = PromptTemplate(
    template =  'give the short 4 line summary of following topic {topic}',
    input_variables = ['topic']
)



summary_chain = summary_prompt | model | parser



chain = RunnableParallel(
    original = RunnablePassthrough(),
    summary = summary_chain
)


result = chain.invoke({'topic': 'cricket'})
print('original: ')
print(result['original'])
print('summary: ')
print(result['summary'])