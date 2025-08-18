import chromadb
import uuid
import json

OPEN_AI_ROLE_MAPPING = {"human": "user", "ai": "assistant"}

class Reflection:
    def __init__(self, llm, db_path: str, dbChatHistoryCollection: str, semanticCacheCollection: str):
        self.client = chromadb.PersistentClient(path=db_path)
        self.his_collection = self.client.get_or_create_collection(name=dbChatHistoryCollection)
        self.semantic_cache_collection = self.client.get_or_create_collection(name=semanticCacheCollection)
        self.llm = llm

    def chat(self, session_id: str, enhanced_message: str, original_message: str = '', cache_response: bool = False, query_embedding: list = None):
        # Build full prompt with context
        system_prompt_content = """Bạn là chatbot cửa hàng bán điện thoại/laptop. Vai trò của bạn là hỗ trợ khách hàng trong việc tìm hiểu về các sản phẩm và dịch vụ của cửa hàng, cũng như tạo một trải nghiệm mua sắm dễ chịu và thân thiện. Bạn có thể trả lời các câu hỏi về loại hoa, dịch vụ giao hàng. Bạn cũng có thể trò chuyện với khách hàng về các chủ đề không liên quan đến sản phẩm như thời tiết, sở thích cá nhân, và những câu chuyện thú vị để tạo sự gắn kết. 
Hãy luôn giữ thái độ lịch sự và chuyên nghiệp. Nếu khách hàng hỏi về sản phẩm cụ thể, hãy cung cấp thông tin chi tiết và gợi ý các lựa chọn phù hợp. Nếu khách hàng trò chuyện về các chủ đề không liên quan đến sản phẩm, hãy tham gia vào cuộc trò chuyện một cách vui vẻ và thân thiện.
một số điểm chính bạn cần lưu ý:
1. Đáp ứng nhanh chóng và chính xác, sử dụng xưng hô là "Mình và bạn".
2. Giữ cho cuộc trò chuyện vui vẻ và thân thiện.
3. Cung cấp thông tin hữu ích về tiệm bánh và dịch vụ của cửa hàng.
4. Giữ cho cuộc trò chuyện mang tính chất hỗ trợ và giúp đỡ.
Hãy làm cho khách hàng cảm thấy được chào đón và quan tâm!"""
        system_prompt = [{"role": "system", "content": system_prompt_content}]
        session_msgs = self.__construct_session_messages__(session_id)
        user_prompt = [{"role": "user", "content": enhanced_message}]
        messages = system_prompt + session_msgs + user_prompt

        response_text = self.llm.chat(messages)

        # Lưu history
        self.__record_human_prompt__(session_id, enhanced_message, original_message)
        self.__record_ai_response__(session_id, response_text)

        # Cache nếu cần
        if cache_response and query_embedding:
            self.__cache_ai_response__(enhanced_message, original_message, response_text, query_embedding)

        return response_text

    def __construct_session_messages__(self, session_id: str):
        session_messages = self.his_collection.get(where_document={"$contains": session_id})
        result = []
        if not session_messages.get('ids'):
            return result
        for doc in session_messages.get('documents', []):
            hist = json.loads(doc).get('History', {})
            role = OPEN_AI_ROLE_MAPPING.get(hist.get('type', 'human'), 'user')
            content = hist.get('data', {}).get('content', '')
            result.append({"role": role, "content": content})
        return result

    def __record_human_prompt__(self, session_id, enhanced_message, original_message):
        self.his_collection.add(
            ids=[str(uuid.uuid4())],
            documents=[json.dumps({
                "SessionId": session_id,
                "History": {
                    "type": "human",
                    "data": {
                        "type": "human",
                        "content": original_message,
                        "enhanced_content": enhanced_message
                    }
                }
            })]
        )

    def __record_ai_response__(self, session_id, response_text):
        self.his_collection.add(
            ids=[str(uuid.uuid4())],
            documents=[json.dumps({
                "SessionId": session_id,
                "History": {
                    "type": "ai",
                    "data": {"type": "ai", "content": response_text}
                }
            })]
        )

    def __cache_ai_response__(self, enhanced_message, original_message, response_text, query_embedding):
        self.semantic_cache_collection.add(
            ids=[str(uuid.uuid4())],
            embeddings=[query_embedding],
            documents=[json.dumps({
                "text": [{"type": "human", "content": original_message, "enhanced_content": enhanced_message}],
                "llm_string": {"model_name": "gpt-4o-mini", "name": "ChatOpenAI"},
                "return_val": [{"type": "ai", "content": response_text}]
            })]
        )
