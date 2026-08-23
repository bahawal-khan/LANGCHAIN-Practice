from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)


class Student(BaseModel):
    name: str = Field(description= 'Student name')
    age: int = Field(description= 'student age')
    city: str = Field(description="Student ka city")
    course: str = Field(description="Student ka course")


parser = PydanticOutputParser(pydantic_object=Student)


prompt = PromptTemplate(
    template="""
    Extract the student information from the following text.{format_instructions}
Student: {student}""",
input_variables=['student'],
partial_variables= {
    'format_instructions': parser.get_format_instructions()
}

)


chain = prompt|model|parser


result = chain.invoke({
    "student": """
    Ali is 20 years old.
    He lives in Lahore.
    He studies Computer Science.
    """
})


print(result)
print(type(result))