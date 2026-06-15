from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

class MovieReview(BaseModel):
    title: str
    sentiment: str    # positive / negative / mixed
    rating: int       # out of 10
    summary: str      # one-line takeaway from the review
    recommended: bool

model = ChatAnthropic(model="claude-haiku-4-5")
structured_model = model.with_structured_output(MovieReview)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a movie review analyzer. Extract structured information from the given movie review."),
    ("human", "{review}"),
])

chain = prompt | structured_model

if __name__ == "__main__":
    review_text = input("Enter your movie review: ").strip()

    result = chain.invoke({"review": review_text})

    print("\n--- Structured Review Output ---")
    print(f"  Title      : {result.title}")
    print(f"  Sentiment  : {result.sentiment}")
    print(f"  Rating     : {result.rating} / 10")
    print(f"  Summary    : {result.summary}")
    print(f"  Recommended: {'Yes' if result.recommended else 'No'}")
