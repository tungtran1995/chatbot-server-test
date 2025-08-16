import chromadb
import json
import uuid
import re
from chromadb.utils import embedding_functions
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ---- EMBEDDING CONFIG ----  
AZURE_OPENAI_EMBEDDING_API_KEY = os.getenv("OPENAI_API_KEY_EMBEDDED")  
AZURE_OPENAI_EMBEDDING_ENDPOINT = os.getenv("OPENAI_ENDPOINT")  
AZURE_OPENAI_EMBED_MODEL = os.getenv("AZURE_OPENAI_EMBED_MODEL", "text-embedding-3-small")

# Đọc dữ liệu sản phẩm từ file JSON
with open('data.json', 'r', encoding='utf-8') as f:
    products = json.load(f)

# Khởi tạo client cho Azure OpenAI Embedding
embedding_client = OpenAI(  
    api_key=AZURE_OPENAI_EMBEDDING_API_KEY,  
    azure_endpoint=AZURE_OPENAI_EMBEDDING_ENDPOINT,  
    api_version="2023-05-15"  
)  

chroma_client = chromadb.PersistentClient(path='VECTOR_STORE')
collection = chroma_client.get_or_create_collection(name="product")

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s,.]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_embedding(text: str):
    response = embedding_client.embeddings.create(
        model=AZURE_OPENAI_EMBED_MODEL,
        input=text
    )
    return response.data[0].embedding

def add_product_to_chromadb(product, product_id, collection):
    required_fields = ["title", "price", "image_url"]
    if not all(field in product and product[field] for field in required_fields):
        print(f"Product {product_id} missing required fields.")
        return
    if collection is None:
        print("Collection is None.")
        return

    # Nếu description rỗng, thay bằng category hoặc title
    description = product.get("description") or product.get("category") or product.get("title")
    combined_text = f"{product['title']} {description}"
    combined_text = preprocess_text(combined_text)

    try:
        embedding = get_embedding(combined_text)
        collection.add(
            ids=[str(product_id)],
            documents=[combined_text],
            embeddings=[embedding],   # <<--- dùng embedding_client tạo
            metadatas=[{
                "title": product["title"],
                "description": description,
                "price": product["price"],
                "image_url": product["image_url"],
                "category": product.get("category", "")
            }]
        )
    except Exception as e:
        print(f"Error adding product {product_id}: {e}")

for item in products:
    product = {
        "title": item.get("name", ""),
        "description": item.get("Description", ""),
        "price": item.get("price", ""),
        "image_url": item.get("picture", ""),
        "category": item.get("category", "")
    }
    product_id = uuid.uuid4()
    add_product_to_chromadb(product, product_id=product_id, collection=collection)
