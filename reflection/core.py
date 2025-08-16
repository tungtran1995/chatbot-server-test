import json
import uuid
import chromadb
from dotenv import load_dotenv

load_dotenv()

OPEN_AI_ROLE_MAPPING = {
    "human": "user",
    "ai": "assistant"
}

class Reflection:
    def __init__(self, llm, db_path: str, dbChatHistoryCollection: str, semanticCacheCollection: str):
        """
        llm: instance của OpenAiClient (chat() trả về string)
        db_path: thư mục lưu trữ dữ liệu ChromaDB
        dbChatHistoryCollection: tên collection lưu lịch sử chat
        semanticCacheCollection: tên collection lưu cache semantic
        cache_ttl: thời gian sống của cache (giây)
        """
        self.client = chromadb.PersistentClient(path=db_path)
        self.his_collection = self.client.get_or_create_collection(name=dbChatHistoryCollection)
        self.semantic_cache_collection = self.client.get_or_create_collection(name=semanticCacheCollection)
        self.llm = llm
        #self.cache_ttl = cache_ttl

    def chat(self, session_id: str, enhanced_message: str, original_message: str = '', cache_response: bool = False, query_embedding: list = None):
        print('call')
        """
        Thực hiện chat với LLM và lưu lịch sử, cache
        """
        system_prompt_content = """Bạn là một chatbot của cửa hàng bán laptop và điện thoại. 
        Vai trò của bạn là hỗ trợ khách hàng trong việc tìm hiểu về các sản phẩm và dịch vụ của cửa hàng, 
        cũng như tạo một trải nghiệm mua sắm dễ chịu và thân thiện."""

        system_prompt = [{"role": "system", "content": system_prompt_content}]
        human_prompt = [{"role": "user", "content": enhanced_message}]

        # Lấy lịch sử session
        session_messages = self.his_collection.get(where_document={"$contains": session_id})
        formatted_session_messages = self.__construct_session_messages__(session_messages)

        # Kết hợp system + session + human prompt
        messages = system_prompt + formatted_session_messages + human_prompt
        print(f"final messages: {messages}")

        # Gọi OpenAiClient chat() trả về string trực tiếp
        response_text = self.llm.chat(messages)

        # Lưu lịch sử user + AI
        self.__record_human_prompt__(session_id, enhanced_message, original_message)
        self.__record_ai_response__(session_id, response_text)

        # Cache nếu có embedding
        if cache_response and query_embedding is not None:
            self.__cache_ai_response__(enhanced_message, original_message, response_text, query_embedding)

        return response_text

    def __construct_session_messages__(self, session_messages: dict):
        """
        Chuyển session_messages từ ChromaDB thành định dạng messages cho LLM
        """
        result = []
        if not session_messages.get('ids'):
            return result

        for session_message in session_messages.get('documents', []):
            session_message = json.loads(session_message)
            hist = session_message.get('History', {})
            role = OPEN_AI_ROLE_MAPPING.get(hist.get('type', 'human'), 'user')
            content = hist.get('data', {}).get('content', '')
            result.append({"role": role, "content": content})

        return result

    def __record_human_prompt__(self, session_id: str, enhanced_message: str, original_message: str):
        """
        Lưu message của user vào collection
        """
        self.his_collection.add(
            ids=[str(uuid.uuid4())],
            documents=[json.dumps({
                "SessionId": session_id,
                "History": {
                    "type": "human",
                    "data": {
                        "type": "human",
                        "content": original_message,
                        "enhanced_content": enhanced_message,
                        "additional_kwargs": {},
                        "response_metadata": {},
                        "name": None,
                        "id": None
                    }
                }
            })]
        )

    def __record_ai_response__(self, session_id: str, response_text: str):
        """
        Lưu message của AI vào collection
        """
        self.his_collection.add(
            ids=[str(uuid.uuid4())],
            documents=[json.dumps({
                "SessionId": session_id,
                "History": {
                    "type": "ai",
                    "data": {
                        "type": "ai",
                        "content": response_text,
                        "enhanced_content": None,
                        "additional_kwargs": {},
                        "name": None,
                        "id": None,
                        "usage_metadata": {},
                        "response_metadata": {}
                    }
                }
            })]
        )

    def __cache_ai_response__(self, enhanced_message: str, original_message: str, response_text: str, query_embedding: list):
        """
        Lưu response + embedding vào semantic cache
        """
        self.semantic_cache_collection.add(
            ids=[str(uuid.uuid4())],
            embeddings=[query_embedding],
            documents=[json.dumps({
                "text": [
                    {
                        "type": "human",
                        "content": original_message,
                        "enhanced_content": enhanced_message,
                        "additional_kwargs": {},
                        "response_metadata": {},
                        "name": None,
                        "id": None
                    }
                ],
                "llm_string": {
                    "model_name": "gpt-4o-mini",
                    "name": "ChatOpenAI"
                },
                "return_val": [
                    {
                        "type": "ai",
                        "content": response_text,
                        "enhanced_content": None,
                        "additional_kwargs": {},
                        "name": None,
                        "id": None,
                        "usage_metadata": {},
                        "response_metadata": {}
                    }
                ]
            })]
        )
