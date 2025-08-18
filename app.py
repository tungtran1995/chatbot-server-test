import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv
from rag import RAG
from reflection import Reflection
from agent_router import GuardedRAGAgent
from openai_client import OpenAiClient

# ===== Load env =====
load_dotenv()

# ===== Kiểm tra env =====
DB_PATH = "VECTOR_STORE"
COLLECTION_NAME = os.getenv("COLLECTION_NAME") or "products"
DB_CHAT_HISTORY_COLLECTION = os.getenv("DB_CHAT_HISTORY_COLLECTION") or "chat_history"
SEMANTIC_CACHE_COLLECTION = os.getenv("SEMANTIC_CACHE_COLLECTION") or "semantic_cache"
EMBED_MODEL = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

for var_name, var_value in [
    ("COLLECTION_NAME", COLLECTION_NAME),
    ("DB_CHAT_HISTORY_COLLECTION", DB_CHAT_HISTORY_COLLECTION),
    ("SEMANTIC_CACHE_COLLECTION", SEMANTIC_CACHE_COLLECTION)
]:
    if not var_value:
        raise ValueError(f"{var_name} chưa được định nghĩa trong .env")

MAX_HISTORY_ITEMS = 100
SIMILARITY_THRESHOLD = 0.75

# ===== Flask app =====
app = Flask(__name__)
CORS(app)

# ===== Embedding client =====
embedding_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY_EMBEDDED"),
    base_url=os.getenv("OPENAI_ENDPOINT")
)

# ===== RAG object =====
rag = RAG(collection_name=COLLECTION_NAME, db_path=DB_PATH)

# ===== Reflection fallback =====
llm_client = OpenAiClient(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_ENDPOINT")
)
reflection = Reflection(
    llm=llm_client,
    db_path=DB_PATH,
    dbChatHistoryCollection=DB_CHAT_HISTORY_COLLECTION,
    semanticCacheCollection=SEMANTIC_CACHE_COLLECTION
)

# ===== Guarded RAG Agent =====
agent_router = GuardedRAGAgent(
    rag=rag,
    embedding_client=embedding_client,
    embed_model=EMBED_MODEL,
    fallback_reflection=reflection,
    similarity_threshold=SIMILARITY_THRESHOLD,
    max_last_items=MAX_HISTORY_ITEMS
)

# ===== API endpoint: chatbot multi-turn =====
@app.route("/api/v1/chatbot", methods=["POST"])
def chatbot():
    data = request.get_json()
    query = data.get("query", "")
    session_id = data.get("session_id", "")

    print(f"[API DEBUG] Incoming query: {query}, session_id: {session_id}")

    # Gọi agent invoke (multi-turn + query rewrite + RAG + fallback)
    result = agent_router.invoke(query=query, session_id=session_id)

    # Debug chi tiết
    print(f"[API DEBUG] Agent output (first 300 chars): {result['output'][:300]}")
    if hasattr(agent_router, 'last_rewritten_query'):
        print(f"[API DEBUG] Rewritten standalone query: {agent_router.last_rewritten_query}")

    return jsonify({"role": "assistant", "content": result["output"]})

# ===== API endpoint: test RAG retrieval =====
@app.route("/api/v1/rag_test", methods=["POST"])
def rag_test():
    data = request.get_json()
    query = data.get("query", "")

    print(f"[API DEBUG] RAG test query: {query}")

    # Lấy embedding cho query
    query_embedding = embedding_client.embeddings.create(
        model=EMBED_MODEL,
        input=query
    ).data[0].embedding

    # Lấy document từ RAG
    results = rag.hybrid_search(query_embedding, limit=5)
    print(f"[DEBUG] Retrieved {len(results)} documents from RAG:")
    for r in results:
        print(f"  _id={r['_id']}, title={r['title']}, distance={r['distance']:.4f}")

    if not results:
        return jsonify({"status": "empty", "message": "Không tìm thấy dữ liệu", "results": []})

    return jsonify({"status": "ok", "message": f"Tìm thấy {len(results)} document", "results": results})

# ===== Run server =====
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
