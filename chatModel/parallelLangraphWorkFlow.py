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
    feedback: str = Field(description="Feedback for the essay in 2-3 sentences")
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
    prompt = f"Evaluate the language quality of the following UPSC essay. Check grammar, vocabulary, sentence structure and tone. Give feedback in 2-3 sentences and a score out of 10. Essay: {state['essay']}"

    output = structured_model.invoke(prompt)

    return {
        "languageFeedback": output.feedback,
        "individualSectionScores": [output.score],
    }

def content_feedback(state: UPSCState):
    prompt = f"Evaluate the content quality of the following UPSC essay. Check depth of analysis, relevance to the topic, use of facts and examples, and balance of arguments. Give feedback in 2-3 sentences and a score out of 10. Essay: {state['essay']}"

    output = structured_model.invoke(prompt)

    return {
        "contentFeedback": output.feedback,
        "individualSectionScores": [output.score],
    }

def clarity_feedback(state: UPSCState):
    prompt = f"Evaluate the clarity of thought of the following UPSC essay. Check logical flow, structure (introduction, body, conclusion), coherence between paragraphs and how clearly the main idea is expressed. Give feedback in 2-3 sentences and a score out of 10. Essay: {state['essay']}"

    output = structured_model.invoke(prompt)

    return {
        "clarityFeedback": output.feedback,
        "individualSectionScores": [output.score],
    }

def overall_feedback(state: UPSCState):
    average_score = sum(state["individualSectionScores"]) / len(state["individualSectionScores"])
    prompt = f"Based on the following feedback and scores, provide an overall evaluation of the UPSC essay. Essay: {state['essay']}. Language Feedback: {state['languageFeedback']}. Content Feedback: {state['contentFeedback']}. Clarity Feedback: {state['clarityFeedback']}. Average Score: {average_score:.2f}/10. Give overall feedback in 2-3 sentences in plain text, without headings or tables. Do not add statistics that are not in the essay."

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
workflow = graph.compile()

essay = """Digital India: Bridging the Gap or Widening It?

The Digital India programme, launched in 2015, aims to transform India into a digitally empowered society. Initiatives such as UPI, Aadhaar and BharatNet have brought banking, identity and internet access to millions of citizens. Today, a street vendor can accept digital payments, and a farmer can check crop prices on a mobile phone.

However, the digital divide remains a serious concern. Rural areas still face poor connectivity, and many women and elderly citizens lack digital literacy. During the COVID-19 pandemic, students without smartphones or internet access fell behind in online education. Rising cyber fraud also threatens trust in digital systems.

To make Digital India truly inclusive, the government must invest in rural infrastructure, promote digital literacy in local languages and strengthen data protection laws. Technology should be a bridge that connects every citizen, not a wall that separates the connected from the unconnected."""

result = workflow.invoke({"essay": essay})

print("Language Feedback:\n", result["languageFeedback"], "\n")
print("Content Feedback:\n", result["contentFeedback"], "\n")
print("Clarity Feedback:\n", result["clarityFeedback"], "\n")
print("Individual Scores:", result["individualSectionScores"])
print("Average Score:", round(result["averageScore"], 2), "\n")
print("Overall Feedback:\n", result["overallFeedback"])
