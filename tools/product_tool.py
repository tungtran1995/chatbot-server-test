from langchain.tools import tool
from rag import RAG  # import client + RAG
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

AZURE_OPENAI_EMBEDDING_API_KEY = os.getenv("OPENAI_API_KEY_EMBEDDED")
AZURE_OPENAI_EMBEDDING_ENDPOINT = os.getenv("OPENAI_ENDPOINT")
AZURE_OPENAI_EMBED_MODEL = os.getenv("AZURE_OPENAI_EMBED_MODEL", "text-embedding-3-small")

embedding_client = OpenAI(
    api_key=AZURE_OPENAI_EMBEDDING_API_KEY,
    base_url=AZURE_OPENAI_EMBEDDING_ENDPOINT,
)

# Tạo object RAG (nếu chưa có)
rag = RAG(collection_name="YOUR_COLLECTION_NAME", db_path="VECTOR_STORE")

@tool
def product_search(query: str) -> str:
    """
    Single-input tool để tìm sản phẩm từ RAG.
    """
    # tạo embedding
    query_embedding = embedding_client.embeddings.create(
        model=AZURE_OPENAI_EMBED_MODEL,
        input=query
    ).data[0].embedding

    # lấy context từ RAG
    context = rag.enhance_prompt(query_embedding)
    if not context:
        return "Không tìm thấy sản phẩm liên quan."

    # kết hợp prompt
    prompt = f"Hãy trả lời câu hỏi sau dựa trên thông tin sản phẩm:\n{context}\nCâu hỏi: {query}"
    return prompt  # agent sẽ gọi LLM để generate final answer
