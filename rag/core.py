import os
import chromadb
from dotenv import load_dotenv
from chromadb.utils import embedding_functions
load_dotenv()

# Default number of top matches to retrieve from vector search
try:
    DEFAULT_SEARCH_LIMIT = int(os.getenv('DEFAULT_SEARCH_LIMIT', 5))
except (TypeError, ValueError):
    DEFAULT_SEARCH_LIMIT = 5

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="keepitreal/vietnamese-sbert")

class RAG():
    def __init__(self, collection_name: str, db_path: str):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=sentence_transformer_ef
        )

    def weighted_reciprocal_rank(self, doc_lists, weights=None, c=60):
        if not doc_lists:
            return []
        if weights is None:
            weights = [1] * len(doc_lists)
        if len(doc_lists) != len(weights):
            raise ValueError("Number of rank lists must equal the number of weights.")

        # Map _id to document
        id_to_doc = {doc["_id"]: doc for doc_list in doc_lists for doc in doc_list}
        rrf_score = {doc_id: 0.0 for doc_id in id_to_doc}

        for doc_list, weight in zip(doc_lists, weights):
            for rank, doc in enumerate(doc_list, start=1):
                rrf_score[doc["_id"]] += weight * (1 / (rank + c))

        sorted_ids = sorted(rrf_score, key=lambda x: rrf_score[x], reverse=True)
        return [id_to_doc[doc_id] for doc_id in sorted_ids]

    def _format_results(self, results):
        # Defensive: check if results are valid
        if not results or not results.get('ids') or not results['ids'][0]:
            return []
        return [
            {
                "_id": results['ids'][0][i],
                "title": results['metadatas'][0][i].get('title', ''),
                "description": results['metadatas'][0][i].get('description', ''),
                "price": results['metadatas'][0][i].get('price', ''),
                "image_url": results['metadatas'][0][i].get('image_url', ''),
                "category": results['metadatas'][0][i].get('category', ''),
                "distance": results['distances'][0][i]
            }
            for i in range(len(results['ids'][0]))
        ]

    def hybrid_search(self, query: str, query_embedding: list, limit=DEFAULT_SEARCH_LIMIT, weights=None):
        if not query or query_embedding is None:
            return []

        # Vector search
        vector_results_raw = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit
        )
        vector_results = self._format_results(vector_results_raw)

        # Keyword search
        keyword_results_raw = self.collection.query(
            query_texts=[query],
            n_results=limit
        )
        keyword_results = self._format_results(keyword_results_raw)

        # Rank fusion
        doc_lists = [vector_results, keyword_results]
        fused_documents = self.weighted_reciprocal_rank(doc_lists, weights=weights)
        return fused_documents

    def enhance_prompt(self, query: str, query_embedding: list):
        get_knowledge = self.hybrid_search(query, query_embedding)
        enhanced_prompt = "\n".join([
            f"Title: {result.get('title', 'N/A')}, Content: {result.get('description', 'N/A')}, Price: {result.get('price', 'N/A')}, Image URLs: {result.get('image_url', 'N/A')}"
            for result in get_knowledge
        ])
        return enhanced_prompt