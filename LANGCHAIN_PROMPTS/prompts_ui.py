# from langchain_groq import ChatGroq
# from dotenv import load_dotenv
# import streamlit as st

# load_dotenv()

# model = ChatGroq(model="openai/gpt-oss-20b")

# st.header("Research Tool")

# user_input = st.text_input("Enter your text:")

# if st.button("Summarize"):

#     ## static prompting
    
#     prompt = f"""
#     Summarize the following text in simple and concise words:

#     {user_input}
#     """

#     result = model.invoke(prompt)

#     st.write(result.content)


from langchain_groq import ChatGroq
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-20b")

st.header("Research Tool")

user_input = st.text_input("Enter your text:")

style = st.selectbox(
    "Choose summary style:",
    ["Simple", "Detailed", "Bullet Points"]
)

if st.button("Summarize"):

    if style == "Simple":
        prompt = f"Summarize this text in simple words:\n{user_input}"

    elif style == "Detailed":
        prompt = f"Give a detailed summary of this text:\n{user_input}"

    else:
        prompt = f"Summarize this text using bullet points:\n{user_input}"

    result = model.invoke(prompt)

    st.write(result.content)