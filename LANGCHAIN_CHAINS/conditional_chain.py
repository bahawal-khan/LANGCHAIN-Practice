from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.runnables import RunnableBranch

load_dotenv()
model = ChatGroq(model = 'openai/gpt-oss-20b')

parser = StrOutputParser()


english_prompt = PromptTemplate(
    template = 'Translate the following text into urdu:\n{text}',
    input_variables=['text']
)

english_chain =  english_prompt | model|parser


urdu_prompt = PromptTemplate(
    template = 'Translate the following text into english:\n{text}',
    input_variables=['text']
)

urdu_chain = urdu_prompt | model | parser

conditional_chain = RunnableBranch(
    (lambda x: x['language'] =='english',english_chain),
    (lambda x: x['language'] == 'urdu',urdu_chain),
    english_chain ##default
)


result = conditional_chain.invoke({
    'language': 'urdu',
    "text":  'مصنوعی ذہانت دنیا بدل رہی ہے'
})

print(result)



