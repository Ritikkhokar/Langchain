from langchain_core.runnables import RunnablePassthrough, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

# ── ASCII Diagram ───────────────────────────────────────────────
DIAGRAM = """
         INPUT (customer feedback text)
                      |
                      ▼
      ┌─────────────────────────────┐
      │     Sentiment Analysis      │
      │  ChatPromptTemplate + LLM   │
      │      + StrOutputParser      │
      └─────────────────────────────┘
                      |
          "positive" or "negative"
                      |
                      ▼
      ┌─────────────────────────────┐
      │   RunnablePassthrough       │
      │   .assign(sentiment=...)    │
      │ adds sentiment to input dict│
      └─────────────────────────────┘
                      |
          {feedback: ..., sentiment: ...}
                      |
                      ▼
             [RunnableBranch]
            /                \\
       "positive"         "negative"
           |                   |
           ▼                   ▼
   [Positive Prompt]   [Negative Prompt]
    (Thank you msg)    (Apology msg)
           |                   |
           ▼                   ▼
       [Anthropic]         [Anthropic]
           |                   |
           ▼                   ▼
   [StrOutputParser]   [StrOutputParser]
            \\               /
              ──────┬──────
                    ▼
           2-LINE CUSTOMER MESSAGE
"""

# ── Model ───────────────────────────────────────────────────────

model = ChatAnthropic(model="claude-haiku-4-5")
str_parser = StrOutputParser()

# ── Pydantic Schema for Sentiment ───────────────────────────────
# Literal["positive", "negative"] enforces ONLY these two values
# Pydantic will raise a validation error if the model tries to return anything else

class SentimentOutput(BaseModel):
    sentiment: Literal["positive", "negative"]

sentiment_parser = PydanticOutputParser(pydantic_object=SentimentOutput)

# ── Prompts ─────────────────────────────────────────────────────

sentiment_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a sentiment analyzer. Analyze the customer feedback.\n\n{format_instructions}"),
    ("human", "{feedback}"),
]).partial(format_instructions=sentiment_parser.get_format_instructions())

positive_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a customer service agent. Write a warm and genuine 2-line thank you message for a happy customer. Keep it short."),
    ("human", "Customer feedback: {feedback}"),
])

negative_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a customer service agent. Write a sincere 2-line apology and improvement assurance for an unhappy customer. Keep it short."),
    ("human", "Customer feedback: {feedback}"),
])

# ── Sub-chains ──────────────────────────────────────────────────

positive_chain = positive_prompt | model | str_parser
negative_chain = negative_prompt | model | str_parser

# ── Conditional Chain ───────────────────────────────────────────

# sentiment_prompt | model | sentiment_parser  → returns SentimentOutput object
# RunnableLambda extracts just the .sentiment string from that object
# So add_sentiment assigns a clean "positive" or "negative" string — nothing else possible
sentiment_chain = (
    sentiment_prompt
    | model
    | sentiment_parser
    | RunnableLambda(lambda x: x.sentiment)
)

add_sentiment = RunnablePassthrough.assign(sentiment=sentiment_chain)

# RunnableBranch checks conditions top-to-bottom and runs the first matching chain.
# Each condition is a lambda that receives the full dict {"feedback": ..., "sentiment": ...}
branch = RunnableBranch(
    (lambda x: "positive" in x["sentiment"].lower(), positive_chain),
    (lambda x: "negative" in x["sentiment"].lower(), negative_chain),
    negative_chain,  # default fallback
)

# Full chain: assign sentiment → branch → 2-line message
full_chain = add_sentiment | branch

# ── Run ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(DIAGRAM)

    feedbacks = [
        "I absolutely loved the product! The delivery was super fast and the quality exceeded my expectations. Highly recommend! Very disappointed. The item arrived damaged and customer support took 5 days to respond. Waste of money.",
    ]

    for feedback in feedbacks:
        print("=" * 60)
        print(f"Feedback : {feedback}")
        print("-" * 60)

        # Run sentiment step separately to show intermediate output
        with_sentiment = add_sentiment.invoke({"feedback": feedback})
        print(f"Sentiment: {with_sentiment['sentiment'].strip()}")
        print("-" * 60)

        # Run branch with the result
        response = branch.invoke(with_sentiment)
        print(f"Message  :\n{response}")

    print("=" * 60)
