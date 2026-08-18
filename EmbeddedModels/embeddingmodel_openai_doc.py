## we can create multiple text embeddings at a time

from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding  = OpenAIEmbeddings(model = 'text-embedding-3-small',dimensions=32)

documents = [
    'LLMs trained on big data',
    'RNN works on sequential data',
    'Transformers most famous paper is Attention all you need'
]

result = embedding.embed_documents(documents)
print(str(result))