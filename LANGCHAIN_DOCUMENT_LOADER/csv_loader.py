from langchain_community.document_loaders import CSVLoader

loader = CSVLoader('customer.csv')

docs = loader.load()

print(len(docs))

print()

print(docs[49])