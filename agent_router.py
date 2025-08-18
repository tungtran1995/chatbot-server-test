import os
from openai_client import OpenAiClient

class GuardedRAGAgent:
    """
    Agent RAG multi-turn với query rewriting / summarization.
    - Dùng chatHistory để tạo câu hỏi standalone.
    - Tìm document RAG dựa trên rewritten query.
    - Fallback Reflection nếu không tìm đủ document.
    """
    def __init__(self, rag, embedding_client, embed_model, fallback_reflection=None, similarity_threshold=0.75, max_last_items=100):
        self.rag = rag
        self.embedding_client = embedding_client
        self.embed_model = embed_model
        self.fallback_reflection = fallback_reflection
        self.similarity_threshold = similarity_threshold
        self.max_last_items = max_last_items
        self.last_rewritten_query = ""

    def is_product_query(self, query: str) -> bool:
        """Check sơ bộ query có liên quan sản phẩm."""
        keywords = ["iphone", "samsung", "laptop", "điện thoại", "máy tính", "máy ảnh"]
        return any(k.lower() in query.lower() for k in keywords)

    def __rewrite_query(self, chatHistory, query):
        """Tạo câu hỏi standalone dựa trên chat history dài hạn."""
        history_to_use = chatHistory[-self.max_last_items:] if len(chatHistory) > self.max_last_items else chatHistory
        historyString = "\n".join([f"{h['role']}: {h['content']}" for h in history_to_use])

        prompt = [{
            "role": "user",
            "content": f"""
Given a chat history and the latest user question, formulate a standalone question in Vietnamese which can be understood without the chat history. 
Chat history:
{historyString}
User question: {query}
Do NOT answer, just rewrite or return the question as-is.
"""
        }]

        rewritten = self.fallback_reflection.llm.chat(prompt)
        print(f"[DEBUG] Rewritten query: {rewritten[:300]}")  # show first 300 chars
        self.last_rewritten_query = rewritten
        return rewritten

    def invoke(self, query: str, session_id: str = ""):
        print(f"[DEBUG] Incoming query: {query}")

        # 1. Nếu query không liên quan sản phẩm
        if not self.is_product_query(query):
            print("[DEBUG] Query không liên quan sản phẩm.")
            if self.fallback_reflection:
                output = self.fallback_reflection.chat(
                    session_id=session_id,
                    enhanced_message=query,
                    original_message=query,
                    cache_response=False
                )
                print(f"[DEBUG] Fallback Reflection output: {output[:200]}...")
                return {"output": output}
            return {"output": "Không tìm thấy dữ liệu"}

        # 2. Lấy toàn bộ chatHistory
        chatHistory = self.fallback_reflection.__construct_session_messages__(session_id) if self.fallback_reflection else []

        # 3. Rewrite query thành standalone
        rewritten_query = self.__rewrite_query(chatHistory, query)

        # 4. Tạo embedding cho rewritten query
        query_embedding = self.embedding_client.embeddings.create(
            model=self.embed_model,
            input=rewritten_query
        ).data[0].embedding

        # 5. Lấy document từ RAG
        results = self.rag.hybrid_search(query_embedding, limit=5)
        print(f"[DEBUG] Retrieved {len(results)} documents from RAG")
        for r in results:
            print(f"  _id={r['_id']}, title={r['title']}, distance={r['distance']:.4f}")

        # 6. Filter theo similarity threshold
        filtered_results = [r for r in results if r['distance'] >= self.similarity_threshold]
        print(f"[DEBUG] Filtered {len(filtered_results)} docs with similarity >= {self.similarity_threshold}")
        for r in filtered_results:
            print(f"  _id={r['_id']}, title={r['title']}, distance={r['distance']:.4f}")

        if not filtered_results:
            print("[DEBUG] Không có document đủ similarity, fallback Reflection.")
            if self.fallback_reflection:
                output = self.fallback_reflection.chat(
                    session_id=session_id,
                    enhanced_message=query,
                    original_message=query,
                    cache_response=False
                )
                print(f"[DEBUG] Fallback Reflection output: {output[:200]}...")
                return {"output": output}
            return {"output": "Không tìm thấy dữ liệu"}

        # 7. Ghép prompt từ các document
        prompt_docs = "\n".join([
            f"Title: {r['title']}, Content: {r['description']}, Price: {r['price']}, Brand: {r['brand']}"
            for r in filtered_results
        ])

        # 8. Tạo message list cho LLM (multi-turn)
        messages = [{"role": "system", "content": "Bạn là chatbot cửa hàng bán điện thoại/laptop, thân thiện."}]
        messages += chatHistory[-self.max_last_items:]  # giữ multi-turn context
        messages.append({"role": "system", "content": f"Thông tin sản phẩm liên quan:\n{prompt_docs}"})
        messages.append({"role": "user", "content": query})

        # 9. Gọi LLM
        llm = OpenAiClient(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_ENDPOINT")
        )
        response = llm.chat(messages)
        print(f"[DEBUG] LLM output (first 300 chars): {response[:300]}")

        # 10. Lưu history
        if self.fallback_reflection:
            self.fallback_reflection.__record_human_prompt__(session_id, query, query)
            self.fallback_reflection.__record_ai_response__(session_id, response)

        return {"output": response}
