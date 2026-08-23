from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model = 'openai/gpt-oss-20b')


messages = [
    SystemMessage(content = 'your are an expert Ai Engineer'),
    HumanMessage(content = 'Tell me about Langchain')

]


result = model.invoke(messages)

messages.append(AIMessage(
    content = result.content
))

for message in messages:

    if isinstance(message, SystemMessage):
        print("System:", message.content)

    elif isinstance(message, HumanMessage):
        print("You:", message.content)

    elif isinstance(message, AIMessage):
        print("AI:", message.content)