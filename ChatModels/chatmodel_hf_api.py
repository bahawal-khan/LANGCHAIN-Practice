from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='deepseek-ai/DeepSeek-V4-Flash-0731',
    task = 'text-generation'
)


model = ChatHuggingFace(llm = llm)
res = model.invoke('who discovered deepseek?')
print(res.content)