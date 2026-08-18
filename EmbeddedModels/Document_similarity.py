
##This is a small project we do a similarity search 

from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

documents = [
    'LLMs trained on big data',
        'RNN works on sequential data',
        'Transformers most famous paper is Attention all you need',
        'CNN works on image based data'

]

query = 'Tell me about transformers'

doc_emb = embedding.embed_documents(documents)

que_emb = embedding.embed_query(query)

##know find the similarity
score = cosine_similarity([que_emb],doc_emb)[0]

print(list(enumerate(score)))