from openai import OpenAI

class OpenAiClient:
    def __init__(self, api_key: str, base_url: str = None):
        if base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=api_key)

    def chat(self, messages, model="gpt-4o-mini"):
        response = self.client.chat.completions.create(
            model=model,
            messages=messages
        )
        # Trả về thẳng string content thay vì object
        return response.choices[0].message.content
