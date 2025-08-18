import chromadb
import numpy as np

DEFAULT_SEARCH_LIMIT = 5

class RAG:
    def __init__(self, collection_name: str, db_path: str):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def _format_results(self, results):
        """Chuyển kết quả từ ChromaDB thành dict dễ dùng."""
        if not results or not results.get("documents") or not results.get("metadatas"):
            return []

        docs_list = results["documents"][0]
        metas_list = results["metadatas"][0]
        ids_list = results["ids"][0] if "ids" in results and results["ids"] else [None]*len(docs_list)
        distances_list = results["distances"][0] if "distances" in results and results["distances"] else [0]*len(docs_list)

        formatted = []
        for i in range(len(docs_list)):
            meta = metas_list[i] if isinstance(metas_list[i], dict) else {}
            formatted.append({
                "_id": ids_list[i],
                "title": meta.get("title") or meta.get("name") or "N/A",
                "description": docs_list[i],
                "price": meta.get("price", "N/A"),
                "brand": meta.get("brand", "N/A"),
                "category": meta.get("tags", "N/A"),
                "distance": distances_list[i]
            })
        return formatted

    def vector_search(self, query_embedding: list, limit=DEFAULT_SEARCH_LIMIT):
        if not query_embedding:
            return []

        results_raw = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit
        )
        results = self._format_results(results_raw)
        print(f"[DEBUG] Vector search results ({len(results)} items):")
        for r in results:
            print(f"  {r['_id']}: {r['title']}, distance={r['distance']:.4f}")
        return results

    def hybrid_search(self, query_embedding: list, limit=DEFAULT_SEARCH_LIMIT):
        vector_results = self.vector_search(query_embedding, limit)
        return vector_results

    def enhance_prompt(self, query_embedding: list):
        results = self.hybrid_search(query_embedding)
        if not results:
            print("[DEBUG] No knowledge retrieved from RAG.")
            return ""

        prompt = "\n".join([
            f"Title: {r['title']}, Content: {r['description']}, Price: {r['price']}, Brand: {r['brand']}"
            for r in results
        ])
        return prompt
