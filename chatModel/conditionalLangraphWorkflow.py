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


# ── Graph ───────────────────────────────────────────────────────


# ── Run ─────────────────────────────────────────────────────────
