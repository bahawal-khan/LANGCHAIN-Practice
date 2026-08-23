from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-20b"
)

# JSON Schema
student_schema = {
    "title": "Student",
    "description": "Information about a student",
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "Student's name"
        },
        "age": {
            "type": ["integer", "null"],
            "description": "Student's age"
        },
        "department": {
            "type": "string",
            "enum": ["AI", "CS", "SE", "DS"],
            "description": "Student's department"
        },
        "cgpa": {
            "type": "number",
            "minimum": 0,
            "maximum": 4,
            "description": "Student's CGPA"
        }
    },
    "required": ["name", "department", "cgpa"]
}


# Structured output
structured_model = model.with_structured_output(
    student_schema
)


student_info = student_info = """
My name is Muhammad Khan and I am 21 years old. I am currently
studying Computer Science at the University of Lahore. My main
area of interest is Artificial Intelligence and Machine Learning.

I have completed courses in Python, Machine Learning, Deep Learning,
Generative AI, and FastAPI. I am currently learning LangChain and
Agentic AI because I want to build intelligent AI applications.

My current CGPA is 3.6. I have worked on several projects including
a Fake News Detection system, an AI chatbot, and a Smart Agriculture
AI application.

For my Fake News Detection project, I used Python, scikit-learn,
TF-IDF, Linear SVM, FastAPI, and React. I also have experience with
PostgreSQL databases and Docker.

My goal is to become an AI Engineer and eventually work on
production-level AI applications. I enjoy solving programming
problems and learning new technologies.
"""


result = structured_model.invoke(
    f"Extract the student's information from this text:\n{student_info}"
)

print(result)