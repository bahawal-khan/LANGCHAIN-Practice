from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)


schema = [
    ResponseSchema(
        name="name",
        description="Student ka name"
    ),
    ResponseSchema(
        name="age",
        description="Student ki age"
    ),
    ResponseSchema(
        name="city",
        description="Student ka city"
    ),
    ResponseSchema(
        name="course",
        description="Student ka course"
    )
]


parser = StructuredOutputParser.from_response_schemas(schema)


prompt = PromptTemplate(
    template="""
    Extract the student information from the following text.

    {format_instructions}

    Student: {student}
    """,

    input_variables=['student'],

    partial_variables={
        'format_instructions': parser.get_format_instructions()
    }
)


chain = prompt | model | parser


result = chain.invoke({
    "student": """
    Ali is 20 years old.
    He lives in Lahore.
    He studies Computer Science.
    """
})

print(result)
print(type(result))


##resonse schema is not use in new lanchain version