from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langchain_core.output_parsers import StrOutputParser
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

# ── Model & Parser ──────────────────────────────────────────────
model = ChatAnthropic(model="claude-haiku-4-5")
str_parser = StrOutputParser()

# ── State ───────────────────────────────────────────────────────
class ReviewState(TypedDict):
    review: str
    sentiment: Literal["positive", "negative"]
    diagnosis: dict
    response: str


# ── Nodes ───────────────────────────────────────────────────────
def findSentiment(state: ReviewState):
    prompt = f"Analyze the sentiment of the following review. Is it positive or negative? Review: {state['review']}"

    output = model.invoke(prompt, output_parser=str_parser)

    return {"sentiment": output.strip().lower()}

# ── Graph ───────────────────────────────────────────────────────
graph = StateGraph(ReviewState)
graph.add_node('find_sentiment', findSentiment)
graph.add_node('run_diagnosis', runDiagnosis)
graph.add_node('negative_response', negativeResponse)
graph.add_node('positive_response', positiveResponse)



# ── Run ─────────────────────────────────────────────────────────
