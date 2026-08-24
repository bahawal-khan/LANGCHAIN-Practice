from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

text = """
Machine learning is a subset of artificial intelligence.
It allows computers to learn patterns from data.
Machine learning is used for prediction and classification.

Deep learning is a type of machine learning.
It uses neural networks with multiple layers.
Deep learning is used in computer vision and NLP.

The solar system contains the Sun and eight planets.
Jupiter is the largest planet in the solar system.
"""


# 1. Embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type='percentile',
    breakpoint_threshold_amount=55
)

chunks = splitter.split_text(text)
print(len(chunks))
print()
print(chunks[0])
print()
print(chunks[1])
print()
print(chunks[2])
print()
print(chunks[3])
