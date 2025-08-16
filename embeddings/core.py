from sentence_transformers import SentenceTransformer

class EmbeddingModel():
    def __init__(self):
        self.embeddingModel = SentenceTransformer('keepitreal/vietnamese-sbert')


    def get_embedding(self, text: str):
        if not text.strip():
            return []
        
        embedding = self.embeddingModel.encode(text, convert_to_tensor=True)
        return embedding.tolist()
