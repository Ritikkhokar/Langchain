from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

# ── ASCII Diagram ───────────────────────────────────────────────
DIAGRAM = """
                    START
                      |
                      ▼
      ┌─────────────────────────────┐
      │      create_outline         │
      │   prompt + LLM + parser     │
      │  state["outline"] = ...     │
      └─────────────────────────────┘
                      |
                      ▼
      ┌─────────────────────────────┐
      │       human_review          │
      │   interrupt({...}) ⏸        │
      │  graph PAUSES here and the  │
      │  outline is handed to you   │
      └─────────────────────────────┘
                      |
        Command(resume="approve" | "your edits")
                      |
                      ▼
      ┌─────────────────────────────┐
      │        write_post           │
      │  writes post from outline   │
      │  + whatever you replied     │
      └─────────────────────────────┘
                      |
                      ▼
                     END

Sequential = one edge out of every node. The only branch in the
whole run is human: approve, or send edits back in.
"""

# ── Model ───────────────────────────────────────────────────────

model = ChatAnthropic(model="claude-haiku-4-5")
str_parser = StrOutputParser()

# ── State ───────────────────────────────────────────────────────
# Every node receives this dict and returns a PARTIAL dict.
# LangGraph merges the returned keys into the state (last write wins,
# since none of these keys use a reducer).

class BlogState(TypedDict):
    topic: str
    outline: str
    feedback: str
    post: str

# ── Prompts ─────────────────────────────────────────────────────

outline_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a blog editor. Write a tight 4-point outline. One line per point, no intro, no commentary."),
    ("human", "Topic: {topic}"),
])

post_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a blog writer. Write a short post (~150 words) that follows the outline exactly.\n"
     "If the reviewer left feedback, apply it before writing."),
    ("human", "Topic: {topic}\n\nOutline:\n{outline}\n\nReviewer feedback: {feedback}"),
])

# ── Nodes ───────────────────────────────────────────────────────

def create_outline(state: BlogState) -> BlogState:
    """Node 1 — first node after START."""
    chain = outline_prompt | model | str_parser
    outline = chain.invoke({"topic": state["topic"]})
    return {"outline": outline}

def human_review(state: BlogState) -> BlogState:
    """Node 2 — the pause.

    interrupt() raises GraphInterrupt on its FIRST call: the node stops here,
    the checkpointer saves the state, and .invoke()/.stream() returns with an
    "__interrupt__" key carrying the payload below.

    When you resume with Command(resume=<value>), this same node runs again
    from the top — and this time interrupt() RETURNS <value> instead of raising.
    That is why a node holding an interrupt must be safe to re-run: keep the
    expensive work (LLM calls, writes) in other nodes.
    """
    answer = interrupt({
        "question": "Approve this outline, or type your edits.",
        "outline": state["outline"],
    })

    # "approve" → nothing to apply; anything else is treated as feedback.
    feedback = "" if answer.strip().lower() in {"approve", "ok", "yes", ""} else answer
    return {"feedback": feedback}

def write_post(state: BlogState) -> BlogState:
    """Node 3 — last node before END."""
    chain = post_prompt | model | str_parser
    post = chain.invoke({
        "topic": state["topic"],
        "outline": state["outline"],
        "feedback": state["feedback"] or "none",
    })
    return {"post": post}

# ── Graph ───────────────────────────────────────────────────────

builder = StateGraph(BlogState)

builder.add_node("create_outline", create_outline)
builder.add_node("human_review", human_review)
builder.add_node("write_post", write_post)

# START and END are the built-in virtual nodes — you never define them,
# you just wire real nodes to them.
builder.add_edge(START, "create_outline")
builder.add_edge("create_outline", "human_review")
builder.add_edge("human_review", "write_post")
builder.add_edge("write_post", END)

# A checkpointer is REQUIRED for interrupt(): the pause is stored in it,
# and the thread_id in the config is what you resume against.
# InMemorySaver dies with the process — swap for SqliteSaver/PostgresSaver
# when the pause needs to outlive the run.
checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# ── Run ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(DIAGRAM)

    config = {"configurable": {"thread_id": "blog-1"}}
    topic = "Why small teams ship faster than big ones"

    print("=" * 60)
    print(f"Topic    : {topic}")
    print("-" * 60)

    # First run: START → create_outline → human_review → PAUSE
    state = graph.invoke({"topic": topic}, config=config)

    payload = state["__interrupt__"][0].value
    print("PAUSED at human_review")
    print(f"Question : {payload['question']}")
    print("-" * 60)
    print(payload["outline"])
    print("-" * 60)

    answer = input("Your reply ('approve' or edits): ")

    # Resume: human_review re-runs, interrupt() returns `answer`,
    # then write_post → END.
    state = graph.invoke(Command(resume=answer), config=config)

    print("-" * 60)
    print(f"Feedback : {state['feedback'] or '(approved as-is)'}")
    print("-" * 60)
    print(f"Post     :\n{state['post']}")
    print("=" * 60)
