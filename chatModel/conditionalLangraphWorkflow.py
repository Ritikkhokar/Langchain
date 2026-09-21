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

class SentimentSchema(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description="Sentiment of the review, either positive or negative")

sentiment_model = model.with_structured_output(SentimentSchema)

class DiagnosisSchema(BaseModel):
    issue_type: Literal["billing", "performance", "bug", "support", "other"] = Field(description="The category of issue mentioned in the review")
    tone: Literal["angry", "frustrated", "disappointed", "calm"] = Field(description="The emotional tone expressed by the user")
    urgency: Literal["low", "medium", "high"] = Field(description="How urgent or critical the issue appears to be")

diagnosis_model = model.with_structured_output(DiagnosisSchema)

# ── State ───────────────────────────────────────────────────────
class ReviewState(TypedDict):
    review: str
    sentiment: Literal["positive", "negative"]
    diagnosis: dict
    response: str


# ── Nodes ───────────────────────────────────────────────────────
def findSentiment(state: ReviewState) -> ReviewState:
    prompt = f"Analyze the sentiment of the following review. Is it positive or negative? Review: {state['review']}"

    output = sentiment_model.invoke(prompt)

    return {"sentiment": output.sentiment}

def runDiagnosis(state: ReviewState) -> ReviewState:
    prompt = f"Diagnose the following negative review. Identify the issue type, the tone and the urgency. Review: {state['review']}"

    output = diagnosis_model.invoke(prompt)

    return {"diagnosis": output.model_dump()}

def positiveResponse(state: ReviewState) -> ReviewState:
    prompt = f"Write a warm, appreciative response thanking the customer for the following positive review. Review: {state['review']}"

    output = (model | str_parser).invoke(prompt)

    return {"response": output}

def negativeResponse(state: ReviewState) -> ReviewState:
    diagnosis = state["diagnosis"]
    prompt = (
        f"Write an empathetic customer support response to the following negative review. "
        f"The issue type is '{diagnosis['issue_type']}', the customer's tone is '{diagnosis['tone']}' "
        f"and the urgency is '{diagnosis['urgency']}'. Address the issue directly and match the response "
        f"to the urgency level. Review: {state['review']}"
    )

    output = (model | str_parser).invoke(prompt)

    return {"response": output}

def checkSentiment(state: ReviewState) -> Literal["positive_response", "run_diagnosis"]:
    if state["sentiment"] == "positive":
        return "positive_response"
    else:
        return "run_diagnosis"

# ── Graph ───────────────────────────────────────────────────────
graph = StateGraph(ReviewState)
graph.add_node('find_sentiment', findSentiment)
graph.add_node('run_diagnosis', runDiagnosis)
graph.add_node('negative_response', negativeResponse)
graph.add_node('positive_response', positiveResponse)

graph.add_edge(START, 'find_sentiment')
graph.add_conditional_edges('find_sentiment', checkSentiment)
graph.add_edge('run_diagnosis', 'negative_response')
graph.add_edge('positive_response', END)
graph.add_edge('negative_response', END)



# ── Run ─────────────────────────────────────────────────────────
