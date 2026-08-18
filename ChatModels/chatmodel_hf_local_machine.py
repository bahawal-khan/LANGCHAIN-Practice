
##This code is load open source model in to loacal machine

from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id='deepseek-ai/DeepSeek-V4-Flash-0731',
    task = 'text-generation',

    pipeline_kwargs=dict(
        temperature = 0.5,
        max_new_tokens = 100
    )

)

model = ChatHuggingFace(llm = llm)

res = model.invoke('what is the capital of pakistan?')
print(res.content)