
##groq have no old LLMS it provide only chatmodels openai provide LLMS but paid.
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model = 'openai/gpt-oss-20b')

result = llm.invoke('Hello bhaya kaisay ho? English mn jawab do')
print(result.content)

result1 = llm.invoke('Hello! my name is khan')
print(result1.content)

result2 = llm.invoke('what is my name?')
print(result2.content)