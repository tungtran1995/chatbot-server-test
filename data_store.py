import chromadb
import json
import uuid
import re
import os
from dotenv import load_dotenv
from openai import OpenAI

# ------------------- Load env -------------------
load_dotenv()

AZURE_OPENAI_EMBEDDING_API_KEY = os.getenv("OPENAI_API_KEY_EMBEDDED")
AZURE_OPENAI_EMBEDDING_ENDPOINT = os.getenv("OPENAI_ENDPOINT")
AZURE_OPENAI_EMBED_MODEL = os.getenv("AZURE_OPENAI_EMBED_MODEL", "text-embedding-3-small")

# ------------------- Load product data -------------------
with open('data.json', 'r', encoding='utf-8') as f:
    products = json.load(f)

# ------------------- Khởi tạo client embedding -------------------
embedding_client = OpenAI(
    api_key=AZURE_OPENAI_EMBEDDING_API_KEY,
    base_url=AZURE_OPENAI_EMBEDDING_ENDPOINT,
)

# ------------------- Khởi tạo ChromaDB -------------------
chroma_client = chromadb.PersistentClient(path='VECTOR_STORE')
collection = chroma_client.get_or_create_collection(name="product")

# ------------------- Helper functions -------------------
def preprocess_text(text: str) -> str:
    """Làm sạch text trước khi tạo embedding."""
    text = text.lower()
    text = re.sub(r'[^\w\s,.]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def create_product_text(product: dict) -> str:
    """Tạo text mô tả sản phẩm để embed."""
    return f"""Product Name: {product.get('name', '')}
Brand: {product.get('brand', '')}
Category: {product.get('category', '')}
Color: {product.get('color', '')}
Capacity: {product.get('capacity', '')}
Price: {product.get('price', '')}
Ram: {product.get('ram', '')}
Description: {product.get('description', '')}"""

# ------------------- Chuẩn bị dữ liệu cho batch -------------------
product_texts = [preprocess_text(create_product_text(p)) for p in products]
product_ids = [str(p.get("index", str(uuid.uuid4()))) for p in products]
metadatas = [{
    "title": p.get("name",""),
    "brand": p.get("brand",""),
    "tags": p.get("category",""),
    "price": p.get("price",""),
    "ram": p.get("ram",""),
    "color": p.get("color",""),
    "capacity": p.get("capacity",""),
} for p in products]

# ------------------- Batching -------------------
BATCH_SIZE = 20  # Bạn có thể thay đổi batch size tuỳ theo RAM và API

for i in range(0, len(products), BATCH_SIZE):
    batch_texts = product_texts[i:i+BATCH_SIZE]
    batch_ids = product_ids[i:i+BATCH_SIZE]
    batch_meta = metadatas[i:i+BATCH_SIZE]

    # Tạo embedding cho batch
    response = embedding_client.embeddings.create(
        model=AZURE_OPENAI_EMBED_MODEL,
        input=batch_texts
    )
    embeddings = [e.embedding for e in response.data]

    # Thêm batch vào ChromaDB
    collection.add(
        embeddings=embeddings,
        documents=batch_texts,
        ids=batch_ids,
        metadatas=batch_meta
    )
    print(f"[DEBUG] Added batch {i}-{i+len(batch_texts)-1}")

print(f"✅ Added {len(products)} products to ChromaDB successfully!")
