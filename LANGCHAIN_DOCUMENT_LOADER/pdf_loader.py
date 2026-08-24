from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('solar_system.pdf')

doc = loader.load()

print(doc[0].page_content)
print()
print(doc[0].metadata)