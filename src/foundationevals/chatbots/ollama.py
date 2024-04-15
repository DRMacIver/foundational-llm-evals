from typing import Any

from foundationevals.chatbots.base import Chatbot


class Ollama(Chatbot):
    def new_client(self) -> Any:
        import ollama

        return ollama

    def complete(self):
        # TODO: Ollama doesn't implement max_tokens
        response = self.client.chat(
            model=self.model,
            messages=self.messages,
            options={
                "temperature": self.temperature,
            },
        )["message"]
        return response["content"].strip()
