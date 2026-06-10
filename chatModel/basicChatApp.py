from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import json
import os

load_dotenv()

HISTORY_FILE = os.path.join(os.path.dirname(__file__), "chat_history.txt")

def load_history():
    """Read past messages from file and return as LangChain message objects."""
    if not os.path.exists(HISTORY_FILE):
        return []
    messages = []
    with open(HISTORY_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if data["role"] == "human":
                messages.append(HumanMessage(content=data["content"]))
            elif data["role"] == "ai":
                messages.append(AIMessage(content=data["content"]))
    return messages

def save_message(role: str, content: str):
    """Append a single message to the history file."""
    with open(HISTORY_FILE, "a") as f:
        f.write(json.dumps({"role": role, "content": content}) + "\n")

model = ChatAnthropic(model="claude-haiku-4-5")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. You remember everything the user has told you in previous conversations."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

chain = prompt | model

def chat():
    chat_history = load_history()

    print("=" * 55)
    print("        Welcome to Basic Chat App (LangChain)")
    print("=" * 55)
    if chat_history:
        print(f"  Loaded {len(chat_history)} messages from previous session.")
    else:
        print("  Starting a fresh conversation.")
    print("  Type 'quit' or 'exit' to end the session.")
    print("=" * 55)
    print()

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit", "bye"]:
            print("\nGoodbye! Your chat is saved — I'll remember everything next time.")
            break

        response = chain.invoke({
            "history": chat_history,
            "input": user_input,
        })

        ai_response = response.content

        print(f"\nAI : {ai_response}")
        print("-" * 55)
        print()

        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=ai_response))

        save_message("human", user_input)
        save_message("ai", ai_response)

if __name__ == "__main__":
    chat()
