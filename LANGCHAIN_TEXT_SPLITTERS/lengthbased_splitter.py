from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# text = """
# Artificial Intelligence is a field of computer science.
# Machine learning is a subset of artificial intelligence.
# Deep learning uses neural networks to learn from data.
# Generative AI can create text, images, audio and code.
# """

# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=100,
#     chunk_overlap=20
# )

# chunks = splitter.split_text(text)

# for i, chunk in enumerate(chunks):
#     print(f"Chunk {i + 1}:")
#     print(chunk)
#     print("-" * 40)




    ## pdf chunking

loader = PyPDFLoader('solar_system.pdf')

docs = loader.load()


splitter1 = CharacterTextSplitter(
    separator= " ",
    chunk_size = 50,
    chunk_overlap = 5
 
)


result = splitter1.split_documents(docs)
print(result[0].page_content)
