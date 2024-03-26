from foundationevals.chatbots.base import Chatbot, Message


class GPT(Chatbot):
    def new_client(self):
        from openai import OpenAI

        return OpenAI()

    def complete(self) -> Message:
        completion = (
            self.client.chat.completions.create(
                messages=self.messages,
                model=self.model,
                temperature=self.temperature,
            )
            .choices[0]
            .message
        )

        return {
            "role": "assistant",
            "content": completion.content or "",
        }
