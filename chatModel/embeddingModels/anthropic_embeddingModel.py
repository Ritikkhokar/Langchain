from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

model_name = "sentence-transformers/all-MiniLM-L6-v2"

embedding = HuggingFaceInferenceAPIEmbeddings(
    api_key=os.environ["HUGGINGFACEHUB_API_TOKEN"],
    model_name=model_name,
    # langchain_community still points at the retired api-inference.huggingface.co host;
    # HuggingFace moved Inference API traffic to this router endpoint.
    api_url=f"https://router.huggingface.co/hf-inference/models/{model_name}/pipeline/feature-extraction",
)

result = embedding.embed_query("What is the capital of India?")

print(str(result))
