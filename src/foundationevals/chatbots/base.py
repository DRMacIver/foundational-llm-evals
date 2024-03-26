from typing import TypedDict, Literal, Any, Type, TypeVar
from pydantic import TypeAdapter, ValidationError
import json

T = TypeVar("T")


class Message(TypedDict):
    role: Literal["user", "assistant", "system"]
    content: str


STRUCTURING_PROMPT = """
Please provide a structured representation of your previous
answer as a JSON value matching the following schema: {SCHEMA}.

Don't wrap it in any unnecessary objects. If it's a list, just
return a list. If it's a number, just return a number, etc.
""".replace(
    "\n", " "
).strip()


class Chatbot:
    def __new__(cls, model, **kwargs):
        kwargs["model"] = model
        if model.startswith("claude"):
            from foundationevals.chatbots.claude import Claude

            subclass = Claude
        elif model.startswith("gpt"):
            from foundationevals.chatbots.gpt import GPT

            subclass = GPT
        else:
            from foundationevals.chatbots.ollama import Ollama

            subclass = Ollama

        result = super().__new__(subclass)  # type: ignore
        result.__init__(**kwargs)
        return result

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        messages: list[Message] | None = None,
    ):
        self.model: str = model
        self.messages: list[Message] = [] if messages is None else list(messages)
        self.temperature: float = temperature
        self.max_tokens = max_tokens
        self.__client = None

    @property
    def name(self):
        """A stable name to use for this chatbot instance."""
        return self.model

    def chat(self, message: str) -> str:
        """Send a message to the chatbot and return the response."""
        self.messages.append(
            {
                "role": "user",
                "content": message,
            }
        )
        result = self.complete()
        self.messages.append({"role": "assistant", "content": result})
        return result

    def clone(self, **kwargs) -> "Chatbot":
        """Create a new chatbot instance with the same state."""
        kwargs.setdefault("model", self.model)
        kwargs.setdefault("temperature", self.temperature)
        kwargs.setdefault("messages", self.messages)
        return self.__class__(**kwargs)

    @property
    def client(self):
        if self.__client is None:
            self.__client = self.new_client()
        return self.__client

    def structure(self, target: Type[T]) -> T:
        clone = self.clone()

        adapter = TypeAdapter(target)

        response = clone.chat(
            STRUCTURING_PROMPT.format(SCHEMA=json.dumps(adapter.json_schema()))
        )

        try:
            parsed = json.loads(response)
        except json.JSONDecodeError:
            start = len(response)
            end = 0
            for c in "{[":
                if c in response:
                    start = min(start, response.index(c))
                    break
            for c in "}]":
                if c in response:
                    end = max(end, response.rindex(c) + 1)
                    break

            if start >= end:
                raise

            payload = response[start:end]
            parsed = json.loads(payload)

        try:
            return adapter.validate_python(parsed)
        except ValidationError:
            if isinstance(parsed, dict) and "value" in parsed:
                return adapter.validate_python(parsed["value"])
            else:
                raise

    def new_client(self) -> Any: ...

    def complete(self) -> str:
        """Complete the current conversation with the assistant."""
