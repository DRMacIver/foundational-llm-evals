from foundationevals.chatbots.base import Chatbot, Message
import time


class Claude(Chatbot):
    def new_client(self):
        from anthropic import Anthropic

        return Anthropic()

    def complete(self) -> str:
        from anthropic import InternalServerError

        for retries_left in range(4, -1, -1):
            try:
                return (
                    self.client.messages.create(
                        messages=self.messages,
                        model=self.model,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )
                    .content[0]
                    .text
                )
            except InternalServerError as e:
                if retries_left == 0:
                    raise e
                time.sleep(1)
        assert False
