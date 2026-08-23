from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence

load_dotenv()

model = ChatGroq(model = 'openai/gpt-oss-20b')

parser = StrOutputParser()


prompt1 = PromptTemplate(
    template = 'Give the joke on following {topic}',
    input_variables=  ['topic']
)

prompt2 = PromptTemplate(
    template= 'Give the summary of following {text}',
    input_variables = ['text']
)


chain = RunnableSequence(
    prompt1,
    model,
    parser,
    (lambda joke: {'text': joke}),
    prompt2,
    model,
    parser
 )

result = chain.invoke({'topic': 'cricket'})
print(result)