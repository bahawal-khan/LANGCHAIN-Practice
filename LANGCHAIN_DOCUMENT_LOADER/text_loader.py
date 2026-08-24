from langchain_community.document_loaders import TextLoader
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGroq(model = 'openai/gpt-oss-20b')

parser = StrOutputParser()

prompt = PromptTemplate(
    template= 'write the 2 line summary of the following. {topic}',
    input_variables=['topic']
)


loader = TextLoader('data.txt')

doc = loader.load()

print(doc[0].page_content)
print()
print(type(doc))
print()
print(doc[0])
print()
print(doc[0].metadata)


chain = prompt |  model | parser

result = chain.invoke({'topic': doc[0].page_content})
print(result)