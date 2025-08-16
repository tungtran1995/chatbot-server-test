import numpy as np
from openai import OpenAI

class OpenAIEmbeddingModel:
    def __init__(self, api_key, model_name="text-embedding-ada-002", endpoint=None):
        self.client = OpenAI(api_key=api_key, base_url=endpoint)
        self.model_name = model_name

    def embed_documents(self, texts):
        response = self.client.embeddings.create(input=texts, model=self.model_name)
        return np.array([item.embedding for item in response.data])

    def embed_query(self, text):
        response = self.client.embeddings.create(input=[text], model=self.model_name)
        return np.array(response.data[0].embedding)