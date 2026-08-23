from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.messages import HumanMessage,AIMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

model =  ChatGroq(model = 'openai/gpt-oss-20b')


##chat template

chat_template = ChatPromptTemplate([
    ('system','you are helpful teachear'),
    MessagesPlaceholder(variable_name =  'chat_history'),
    ('human', '{question}')
    
])

# Previous chat history
chat_history = [
    HumanMessage(content="My name is Khan. I am learning AI and Machine Learning."),

    AIMessage(
        content="Nice to meet you, Khan! AI and Machine Learning are great fields to explore."
    ),

    HumanMessage(content="I want to learn about Generative AI."),

    AIMessage(
        content="Generative AI is a type of AI that can create new content such as text, images, audio, and code."
    )
]


##now question 
prompt = chat_template.invoke({
    'chat_history': chat_history,
    'question': 'what i am learning?'
})


##send prompt to LLM

result = model.invoke(prompt)
print('AI: ', result.content)