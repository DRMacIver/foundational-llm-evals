from typing import Any

from foundationevals.chatbots.base import Chatbot


class Ollama(Chatbot):
    def new_client(self) -> Any:
        import ollama

        return ollama

    def complete(self):
        response = self.client.chat(
            model=self.model,
            messages=self.messages,
            options={
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        )["message"]
        return response["content"].strip()
