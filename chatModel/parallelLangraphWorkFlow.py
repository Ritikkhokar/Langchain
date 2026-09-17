from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()





# ── Model & Parser ──────────────────────────────────────────────
model = ChatAnthropic(model="claude-haiku-4-5")
str_parser = StrOutputParser()

class EvaluationSchema(BaseModel):
    feedback: str = Field(description="Detailed feedback for the essay")
    score: int = Field(description="Score out of 10", ge=0, le=10)

structured_model = model.with_structured_output(EvaluationSchema)

# ── State ───────────────────────────────────────────────────────
class UPSCState(TypedDict):
    essay: str
    languageFeedback: str
    contentFeedback: str
    clarityFeedback: str
    overallFeedback: str
    individualSectionScores: Annotated[list[int], operator.add]
    averageScore: float



# ── Nodes ───────────────────────────────────────────────────────
graph = StateGraph(UPSCState)



def language_feedback(state: UPSCState):
    prompt = f"Evaluate the language quality of the following UPSC essay. Check grammar, vocabulary, sentence structure and tone. Give detailed feedback and a score out of 10. Essay: {state['essay']}"

    output = structured_model.invoke(prompt)

    return {
        "languageFeedback": output.feedback,
        "individualSectionScores": [output.score],
    }

def content_feedback(state: UPSCState):
    prompt = f"Evaluate the content quality of the following UPSC essay. Check depth of analysis, relevance to the topic, use of facts and examples, and balance of arguments. Give detailed feedback and a score out of 10. Essay: {state['essay']}"

    output = structured_model.invoke(prompt)

    return {
        "contentFeedback": output.feedback,
        "individualSectionScores": [output.score],
    }

def clarity_feedback(state: UPSCState):
    prompt = f"Evaluate the clarity of thought of the following UPSC essay. Check logical flow, structure (introduction, body, conclusion), coherence between paragraphs and how clearly the main idea is expressed. Give detailed feedback and a score out of 10. Essay: {state['essay']}"

    output = structured_model.invoke(prompt)

    return {
        "clarityFeedback": output.feedback,
        "individualSectionScores": [output.score],
    }


graph.add_node("languageFeedback", language_feedback)
graph.add_node("contentFeedback", content_feedback)
graph.add_node("clarityFeedback", clarity_feedback)


# ── Graph ───────────────────────────────────────────────────────
graph.add_edge(START, "languageFeedback")
graph.add_edge(START, "contentFeedback")
graph.add_edge(START, "clarityFeedback")


# ── Run ─────────────────────────────────────────────────────────
