from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os   
from dotenv import load_dotenv

load_dotenv()

class EmbeddingModel():
    def __init__(self, api_key=None, endpoint=None, model="text-embedding-3-small"):
        """
        api_key: OpenAI API Key hoặc Azure OpenAI Key
        endpoint: Nếu dùng Azure OpenAI thì truyền base_url
        model: model embedding
        """
        self.model = model
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY_EMBEDDED"),
            base_url=endpoint or os.getenv("OPENAI_ENDPOINT")
        )

    def get_embedding(self, text):
        if isinstance(text, list):
            text = " ".join(text)
        if not text.strip():
            return []
        embedding = self.embeddingModel.encode(text, convert_to_tensor=True)
        return embedding.tolist()

