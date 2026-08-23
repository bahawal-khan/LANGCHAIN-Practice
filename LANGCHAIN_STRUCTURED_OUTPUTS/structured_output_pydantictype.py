from langchain_groq import ChatGroq
from pydantic import BaseModel,Field,EmailStr
from typing import Optional,Literal
from dotenv import  load_dotenv

load_dotenv()


model  =  ChatGroq(model = "openai/gpt-oss-20b")


class Student(BaseModel):
    name: str = Field(description='student name')
    age: Optional[int] = Field(default=None,description='student age')
    email:Optional[EmailStr] = Field(default=None,description='student email')
    department: Literal['AI','CS'] = Field(description='Student Department')
    cgpa: float = Field(gt=2,lt = 4,description='student cgpa')


structured_model = model.with_structured_output(Student)

student_info = """
My name is Khan. I am studying AI.
My CGPA is 3.5 and I am 20 years old.
My email is  khan@gmail.com
"""


result = structured_model.invoke(
    f"Extract the student's information from this text:\n{student_info}"
)

print(result)