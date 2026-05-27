from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.9)

# Use invoke() method instead of calling directly
response = llm.invoke("What year did the first man land on the moon?")
print(response)
