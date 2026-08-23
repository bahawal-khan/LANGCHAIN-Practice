from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    task = 'text-generation'
)


model = ChatHuggingFace(llm = llm)

parser = JsonOutputParser()

prompt = PromptTemplate(
    template="""
    Give information about the following student.

    {format_instructions}

    Student: {student}
    """,
    input_variables=  ['Student'],
    partial_variables= {
        'format_instructions': parser.get_format_instructions()
    }
)

chain = prompt | model |  parser

result = chain.invoke({
    "student": "Ali, 20 years old, from Lahore"
})

print(result)
print(type(result))

