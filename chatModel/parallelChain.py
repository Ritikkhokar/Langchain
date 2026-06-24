from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# ── ASCII Diagram ──────────────────────────────────────────────
DIAGRAM = """
              INPUT TEXT
                  |
       ┌──────────┴──────────┐
       │                     │
       ▼                     ▼
  [Prompt 1]            [Prompt 2]
  (Make Notes)          (Make Quiz)
       │                     │
       ▼                     ▼
  [Anthropic            [HuggingFace
  Claude Haiku]         Llama 3.1]
       │                     │
       ▼                     ▼
[StrOutputParser]   [StrOutputParser]
       │                     │
       └──────────┬──────────┘
                  │
                  ▼
             [Prompt 3]
         (Merge Notes + Quiz)
                  │
                  ▼
        [Anthropic Claude Haiku]
                  │
                  ▼
           [StrOutputParser]
                  │
                  ▼
           FINAL DOCUMENT
"""

# ── Models ─────────────────────────────────────────────────────

anthropic_model = ChatAnthropic(model="claude-haiku-4-5")

hf_model = ChatOpenAI(
    model="meta-llama/Llama-3.1-8B-Instruct",
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HUGGINGFACEHUB_API_TOKEN"],
    max_tokens=600,
)

# ── Prompts ────────────────────────────────────────────────────

notes_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a note-taking assistant. Create clear, concise bullet-point notes from the given text. Cover all key points."),
    ("human", "{input}"),
])

quiz_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a quiz creator. Create exactly 3 multiple choice questions (with 4 options each and the correct answer marked) from the given text."),
    ("human", "{input}"),
])

merge_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a document formatter. Combine the notes and quiz into one clean, well-structured study document with clear section headings."),
    ("human", "NOTES:\n{notes}\n\nQUIZ:\n{quiz}"),
])

# ── Chains ─────────────────────────────────────────────────────

str_parser = StrOutputParser()

notes_chain = notes_prompt | anthropic_model | str_parser
quiz_chain  = quiz_prompt  | hf_model        | str_parser

parallel_chain = RunnableParallel(
    notes=notes_chain,
    quiz=quiz_chain,
)

merge_chain = merge_prompt | anthropic_model | str_parser

chain = parallel_chain | merge_chain

# ── Input Text ─────────────────────────────────────────────────

INPUT_TEXT = """
The water cycle, also known as the hydrological cycle, is the continuous movement of water
within Earth and its atmosphere. It involves several key processes: evaporation (water turning
to vapor from oceans, lakes, and rivers), transpiration (plants releasing water vapor),
condensation (water vapor cooling and forming clouds), precipitation (water falling as rain
or snow), and collection (water gathering in oceans, rivers, and groundwater). The sun
provides the energy that drives the cycle, while gravity helps water flow downhill. This cycle
is essential for distributing fresh water around the planet and regulating Earth's temperature.
"""

# ── Run ────────────────────────────────────────────────────────

if __name__ == "__main__":
    # print(DIAGRAM)

    # print("=" * 60)
    # print("STEP 1 & 2 — Running Parallel Chains...")
    # print("=" * 60)

    # parallel_result = parallel_chain.invoke({"input": INPUT_TEXT})

    # print("\n--- NOTES  (Anthropic Claude Haiku + StrOutputParser) ---")
    # print(parallel_result["notes"])

    # print("\n--- QUIZ  (HuggingFace Llama 3.1 + StrOutputParser) ---")
    # print(parallel_result["quiz"])

    # print("\n" + "=" * 60)
    # print("STEP 3 — Merging with Anthropic Claude Haiku...")
    # print("=" * 60)

    final_doc = chain.invoke({"input": INPUT_TEXT})

    print("\n--- FINAL MERGED DOCUMENT  (StrOutputParser) ---")
    print(final_doc)

    chain.get_graph().print_ascii() # Visualize the chain graph
