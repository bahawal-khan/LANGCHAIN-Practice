from langchain_text_splitters  import RecursiveCharacterTextSplitter


text = """
Artificial Intelligence is a field of computer science.
Machine learning is a subset of artificial intelligence.
Deep learning uses neural networks to learn from data.
Generative AI can create text, images, audio and code.
"""


splitter = RecursiveCharacterTextSplitter(
    chunk_size = 50,
    chunk_overlap = 0
)




chunks = splitter.split_text(text)
print(len(chunks))
print()
print(chunks)