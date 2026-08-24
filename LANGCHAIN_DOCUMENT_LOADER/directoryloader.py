from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader

loader = DirectoryLoader(
    'books', glob='*.pdf',loader_cls=PyPDFLoader
)


# docs = loader.load()

# print(len(docs))
# print()
# print(docs[1].page_content)
# print()
# print(docs[1].metadata)


# ## load()

# for doc in docs:
#    print(doc.page_content)


docs = loader.lazy_load()


for doc in docs:
    print(doc.page_content)