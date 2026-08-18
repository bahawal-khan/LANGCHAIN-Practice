from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash-lite')
result = model.invoke('who is the prime minister of pakistan? ')
print(result.content)