from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing import List
import os

load_dotenv()

# --- Step 1: Define the schema ---

class Person(BaseModel):
    name: str       = Field(description="Full name of the person")
    net_worth: str  = Field(description="Net worth in dollars e.g. $200 billion")
    age: int        = Field(description="Current age in years")
    country: str    = Field(description="Country of origin or citizenship")
    height: str     = Field(description="Height e.g. 6 ft 1 in")

class RichestPeopleList(BaseModel):
    people: List[Person] = Field(description="List of the top 5 richest people in the world")


# --- Step 2: Create the parser ---

parser = PydanticOutputParser(pydantic_object=RichestPeopleList)


# --- Step 3: Build the prompt template ---

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a JSON generator. Output ONLY valid JSON. No explanation, no markdown, no code blocks.\n\n{format_instructions}"),
    ("human", "{query}"),
]).partial(format_instructions=parser.get_format_instructions())


# --- Step 4: Set up the HuggingFace model via OpenAI-compatible router ---
# HuggingFace router exposes an OpenAI-compatible API at router.huggingface.co/v1
# so we use ChatOpenAI and just point base_url at HuggingFace instead of OpenAI

model = ChatOpenAI(
    model="meta-llama/Llama-3.1-8B-Instruct",
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HUGGINGFACEHUB_API_TOKEN"],
    max_tokens=800,
)


# --- Step 5: Build the chain ---

chain = prompt | model | parser


# --- Step 6: Run ---

if __name__ == "__main__":
    query = "Who are the top 5 richest people in the world? Include name, net worth, age, country, and height."

    result = chain.invoke({"query": query})

    print("--- Structured Output ---")
    for i, person in enumerate(result.people, 1):
        print(f"\n  #{i}  {person.name}")
        print(f"       Net Worth : {person.net_worth}")
        print(f"       Age       : {person.age}")
        print(f"       Country   : {person.country}")
        print(f"       Height    : {person.height}")
