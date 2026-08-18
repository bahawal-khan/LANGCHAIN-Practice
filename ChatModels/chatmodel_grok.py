
##groq have no old llma it only provide chatmodels
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model = 'openai/gpt-oss-120b',temperature = 1.5,max_tokens=100)

result = model.invoke('hello what is the capital of pakistan')

print(result.content)
