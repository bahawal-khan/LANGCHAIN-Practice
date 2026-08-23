from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()


model = ChatGroq(model = "openai/gpt-oss-20b")
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a {domain} expert"),
    ("human", "Tell me something about {topic}")
])

prompt = chat_template.invoke({
    "domain": "cricket",
    "topic": "dosra"
})

result = model.invoke(prompt)

print(result.content)