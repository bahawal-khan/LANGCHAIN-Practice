from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict,Literal

load_dotenv()

model = ChatGroq(model = 'openai/gpt-oss-20b')


class Review(TypedDict):
    summary: str
    sentiment: Literal['positive','negative','neutral']
    rating: int

structured_model = model.with_structured_output(Review)


review = """
The product is really average. The quality is excellent
and I am very sad with my purchase. I would definitely
say other to first check before purchase.
"""



result = structured_model.invoke(
      f"""
    Analyze this product review.
    Give rating as a whole number from 1 to 5. Do not use decimals.

    Review:
    {review}
    """
)

print(result)