from langchain_text_splitters import RecursiveCharacterTextSplitter,Language


code = """
class Calculator:

    def add(self, a, b):
        return a + b

    def multiply(self, a, b):
        return a * b
"""


splitter = RecursiveCharacterTextSplitter.from_language(
    language= Language.PYTHON,
    chunk_size = 100,
    chunk_overlap = 0
)


chunk = splitter.split_text(code)

print(len(chunk))
print(chunk[0])