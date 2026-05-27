from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

# Initialize ChatOpenAI (for chat models like gpt-3.5-turbo)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.9)

# Method 1: Direct chat message
print("=" * 60)
print("Method 1: Direct Chat Message")
print("=" * 60)
human_message = HumanMessage(content="What year did the first man land on the moon?")
response = llm.invoke([human_message])
print(f"Response: {response.content}\n")

# Method 2: With system message
print("=" * 60)
print("Method 2: With System Message")
print("=" * 60)
system_message = SystemMessage(content="You are a helpful history expert. Answer questions concisely.")
human_message = HumanMessage(content="What year did the first man land on the moon?")
response = llm.invoke([system_message, human_message])
print(f"Response: {response.content}\n")

# Method 3: Using ChatPromptTemplate
print("=" * 60)
print("Method 3: Using ChatPromptTemplate")
print("=" * 60)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    ("human", "{question}")
])

chain = prompt | llm
response = chain.invoke({"question": "What year did the first man land on the moon?"})
print(f"Response: {response.content}\n")

# Method 4: Multi-turn conversation
print("=" * 60)
print("Method 4: Multi-turn Conversation")
print("=" * 60)
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What year did the first man land on the moon?"),
]
response1 = llm.invoke(messages)
print(f"AI: {response1.content}\n")

# Follow-up question
messages.append(response1)
messages.append(HumanMessage(content="Who was the first person?"))
response2 = llm.invoke(messages)
print(f"AI: {response2.content}\n")
