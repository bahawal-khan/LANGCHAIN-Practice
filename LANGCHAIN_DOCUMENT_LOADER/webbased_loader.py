from langchain_community.document_loaders import WebBaseLoader
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

url = 'https://en.wikipedia.org/wiki/Artificial_intelligence'

loader = WebBaseLoader(url)


doc = loader.load()


load_dotenv()


model = ChatGroq(model = 'openai/gpt-oss-20b')

parser = StrOutputParser()

prompt = PromptTemplate(
    template="Summarize the following text in exactly 2 lines:\n\n{text}",
    input_variables=["text"]
)



chain = prompt | model | parser
text = doc[0].page_content[:10000]

result = chain.invoke({'text': text})
print(result)