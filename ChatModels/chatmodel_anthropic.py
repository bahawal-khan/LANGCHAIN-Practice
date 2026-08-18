##Anthropic api keys also paid 
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model = 'claude-sonnet-5')

result = model.invoke('what is the capital of pakistan?')
print(result.content)
