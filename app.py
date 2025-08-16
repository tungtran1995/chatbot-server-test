from flask import Flask, request, jsonify
from flask_cors import CORS
from openai_client import OpenAiClient
from semantic_router.openai_embedding import OpenAIEmbeddingModel
from rag.core import RAG
from semantic_router import SemanticRouter, Route
from semantic_router.samples import productsSample, chitchatSample
from reflection.core import Reflection
import os
from dotenv import load_dotenv
import numpy as np
load_dotenv()

# Load environment variables
api_key=os.getenv('OPENAI_API_KEY')
base_url=os.getenv('OPENAI_ENDPOINT')
db_chat_history_collection=os.getenv('DB_CHAT_HISTORY_COLLECTION')
collection_name=os.getenv('COLLECTION_NAME')
semanticCacheCollection=os.getenv('semanticCacheCollection')
db_path='databaseQQ/VECTOR_STORE'

app = Flask(__name__)
CORS(app)

# Embedding model OpenAI

openai_embed_model = OpenAIEmbeddingModel(
    api_key=os.getenv("OPENAI_API_KEY_EMBEDDED"),
    model_name=os.getenv("OPEN_API_EMBEDDING_MODEL", "text-embedding-ada-002"),
    endpoint=os.getenv("OPENAI_ENDPOINT")
)

rag = RAG(
    collection_name=collection_name,
    db_path=db_path
)

PRODUCT_ROUTE_NAME = 'products'
CHITCHAT_ROUTE_NAME = 'chitchat'
productRoute = Route(name=PRODUCT_ROUTE_NAME, samples=productsSample)
chitchatRoute = Route(name=CHITCHAT_ROUTE_NAME, samples=chitchatSample)
semanticRouter = SemanticRouter(routes=[productRoute, chitchatRoute], embedding=openai_embed_model)

reflection = Reflection(
    llm=OpenAiClient(api_key, base_url),
    db_path=db_path,
    dbChatHistoryCollection=db_chat_history_collection,
    semanticCacheCollection=semanticCacheCollection
)

@app.route('/api/v1/chatbot', methods=['POST'])
def chat():
    data = request.get_json()
    session_id = data.get('session_id', '')
    query = data.get('query', '')

    guided_route = semanticRouter.guide(query)[1]
    print(f"semantic route: {guided_route}")

    if guided_route == PRODUCT_ROUTE_NAME:
        query_embedding = openai_embed_model.embed_query(query)
        source_information = rag.enhance_prompt(query, query_embedding).replace('<br>', '\n')
        combined_information = f"Câu hỏi : {query}, \ntrả lời khách hàng sử dụng thông tin sản phẩm sau:\n###Sản Phẩm###\n{source_information}."
        response = reflection.chat(
            session_id=session_id,
            enhanced_message=combined_information,
            original_message=query,
            cache_response=True,
            query_embedding=query_embedding
        )
    else:
        response = reflection.chat(
            session_id=session_id,
            enhanced_message=query,
            original_message=query,
            cache_response=False
        )

    # lấy content từ OpenAI object
    return jsonify({
        "role": "assistant",
        "content": response
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5001', debug=True)