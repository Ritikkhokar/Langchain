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

def overall_feedback(state: UPSCState):
    average_score = sum(state["individualSectionScores"]) / len(state["individualSectionScores"])
    prompt = f"Based on the following feedback and scores, provide an overall evaluation of the UPSC essay. Essay: {state['essay']}. Language Feedback: {state['languageFeedback']}. Content Feedback: {state['contentFeedback']}. Clarity Feedback: {state['clarityFeedback']}. Average Score: {average_score:.2f}/10. Provide detailed overall feedback."

    output = (model | str_parser).invoke(prompt)

    return {
        "overallFeedback": output,
        "averageScore": average_score,
    }

graph.add_node("languageFeedback", language_feedback)
graph.add_node("contentFeedback", content_feedback)
graph.add_node("clarityFeedback", clarity_feedback)
graph.add_node("overallFeedback", overall_feedback)


# ── Graph ───────────────────────────────────────────────────────
graph.add_edge(START, "languageFeedback")
graph.add_edge(START, "contentFeedback")
graph.add_edge(START, "clarityFeedback")
graph.add_edge("languageFeedback", "overallFeedback")
graph.add_edge("contentFeedback", "overallFeedback")
graph.add_edge("clarityFeedback", "overallFeedback")
graph.add_edge("overallFeedback", END)

# ── Run ─────────────────────────────────────────────────────────
