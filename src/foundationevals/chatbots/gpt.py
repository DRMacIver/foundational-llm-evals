from foundationevals.chatbots.base import Chatbot, Message


class GPT(Chatbot):
    def new_client(self):
        from openai import OpenAI

        return OpenAI()

    def complete(self) -> str:
        completion = (
            self.client.chat.completions.create(
                messages=self.messages,
                model=self.model,
                temperature=self.temperature,
            )
            .choices[0]
            .message
        )

        return completion.content or ""
