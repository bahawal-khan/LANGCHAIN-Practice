from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage,AIMessage
from dotenv import load_dotenv

load_dotenv()


model = ChatGroq(model = 'openai/gpt-oss-20b')

chat_history = []
while True:
    user_input = input('you: ')
    chat_history.append(
        HumanMessage(content = user_input)
    )
    if user_input =='exit':
        break
    result = model.invoke(chat_history)
    chat_history.append(
        AIMessage(content=result.content)
    )
    print('AI: ',result.content)
print('\nChatHistory')
for message in chat_history:
    if isinstance(message,HumanMessage):
        print('you: ',message.content)
    elif isinstance(message,AIMessage):
        print('AI: ',message.content)