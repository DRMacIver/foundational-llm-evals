from typing import TypedDict, Literal, Any, Type, TypeVar
from pydantic import TypeAdapter, ValidationError
import json
import re

T = TypeVar("T")


class Message(TypedDict):
    role: Literal["user", "assistant", "system"]
    content: str


REFUSAL_PHRASES = [
    "i'm sorry",
    "i apologize",
    "it goes against ethical and moral standards",
    "i cannot promote or encourage the use of offensive language",
    "it's important to always use respectful and appropriate language",
    "i cannot fulfill that request",
    "i cannot fulfill your request",
    "i'm just an ai",
    "i cannot provide",
    "my purpose is to assist and provide helpful responses",
]


PERCENTAGE = re.compile(r"([0-9]+(?:\.[0-9]*)?)%")

STRUCTURING_PROMPT = """
Please provide a structured representation of your previous
answer as a JSON value matching the following schema: {SCHEMA}.

Don't wrap it in any unnecessary objects. If it's a list, just
return a list. If it's a number, just return a number, etc.
""".replace(
    "\n", " "
).strip()


class FailedToAnswer(Exception):
    pass


class ResponseParsingError(Exception):
    pass


def extract_json_objects(text: str) -> list[dict[str, Any] | list[Any]]:
    """Extracts JSON objects from a string."""
    try:
        return [json.loads(text)]
    except json.JSONDecodeError:
        pass
    results = []
    stack = []
    for i, c in enumerate(text):
        fragment = None
        if c == '"':
            if stack and stack[-1][0] == '"':
                opener, start = stack.pop()
                fragment = text[start : i + 1]
            else:
                stack.append((c, i))
        elif c in "{[":
            stack.append((c, i))
        elif c in "]}":
            if stack:
                opener, start = stack.pop()
                fragment = text[start : i + 1]
        if fragment and not stack:
            try:
                results.append(json.loads(fragment))
            except json.JSONDecodeError:
                results.extend(extract_json_objects(text[start + 1 : i]))
    return results


def conform_json_to_type(target_type: Type[T], json_object: Any) -> T:
    adapter = TypeAdapter(target_type)
    try:
        return adapter.validate_python(json_object)
    except ValidationError:
        pass

    if isinstance(json_object, dict):
        json_object.pop("type", None)
        if len(json_object) == 1 or (len(json_object) == 2 and "type" in json_object):
            (wrapped,) = [v for k, v in json_object.items() if k != "type"]
            return adapter.validate_python(wrapped)
    if isinstance(json_object, list) and len(json_object) == 1:
        try:
            return conform_json_to_type(target_type, json_object[0])
        except ValidationError:
            pass
    if (
        hasattr(target_type, "__origin__")
        and target_type.__origin__ is list
        and isinstance(json_object, list)
    ):
        result = []
        item_type = target_type.__args__[0]
        for value in json_object:
            try:
                result.append(conform_json_to_type(item_type, value))
            except ValidationError:
                result.extend(conform_json_to_type(target_type, value))
        return adapter.validate_python(result)

    # This is expected to fail, but we want it to fail with an informative message.
    return adapter.validate_python(json_object)


class Chatbot:
    def __new__(cls, model, **kwargs):
        kwargs["model"] = model
        if model == "dummy":
            subclass = DummyChatbot
        elif model.startswith("claude"):
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
        self.children = []
        self.__frozen = False

    @property
    def name(self):
        """A stable name to use for this chatbot instance."""
        return self.model

    def freeze(self):
        self.__frozen = True

    def chat(self, message: str, response: str | None = None) -> str:
        """Send a message to the chatbot and return the response."""
        if self.__frozen:
            raise RuntimeError(
                "Cannot chat with a frozen chatbot. Clone this if you want to chat."
            )
        self.messages.append(
            {
                "role": "user",
                "content": message,
            }
        )
        result = self.complete() if response is None else response
        self.messages.append({"role": "assistant", "content": result})
        return result

    def clone(self, **kwargs) -> "Chatbot":
        """Create a new chatbot instance with the same state."""
        kwargs.setdefault("model", self.model)
        kwargs.setdefault("temperature", self.temperature)
        kwargs.setdefault("messages", self.messages)
        result = self.__class__(**kwargs)
        self.children.append(result)
        return result

    @property
    def client(self):
        if self.__client is None:
            self.__client = self.new_client()
        return self.__client

    def transcript(self) -> str:
        parts = []
        for message in self.messages:
            parts.append(f"{message['role']}: {message['content']}\n")
        return "\n".join(parts)

    @property
    def last_response(self):
        """The last response from the chatbot."""
        assert self.messages and self.messages[-1]["role"] == "assistant"
        return self.messages[-1]["content"]

    def did_the_bot_answer(self) -> bool:
        """Checks whether the last response is a refusal or other
        failure to answer, returning True if the bot believes it
        has actually answered the question."""
        refusal_checking_bot = self.clone()

        check_for_refusal = refusal_checking_bot.chat(
            "Does your answer contain the information I requested? Don't apologise or explain, please just say 'yes' or 'no'"
        ).lower()

        if check_for_refusal.startswith("yes"):
            return True
        elif check_for_refusal.startswith("no"):
            return False
        else:
            return "yes" in check_for_refusal

    def confidence(self) -> float:
        """Return the confidence of the last response, as a float
        between 0 and 1, with 1 indicating certainty and 0 indicating
        that it's certain the answer is wrong."""
        bot = self.clone()
        bot.chat(
            "Give me a number between 0 and 100% that predicts whether. "
            "that was a correct answer to the previous question. "
            "This will be scored based on whether your previous answer "
            "correctly answers the question asked of you. "
            "If it does then scores close to 100% are better. "
            "If it does not then scores closer to 0% are better. "
            "If the answer is definitely correct, please answer 100%. "
            "If the answer is definitely incorrect, please answer 0%. "
            "Please give an answer that will get a high score according to "
            "these rules. There are no wrong answers, only better and worse ones, "
            "so a lack of certainty is not a problem. If you really have no idea, "
            "just answer 50%, but you can probably do better than that. "
            "There are no good reasons to refuse to provide a number - providing any "
            "number is providing better than a refusal to answer."
        )
        return bot.parse_confidence()

    def parse_confidence(self):
        confidence = PERCENTAGE.search(self.last_response)
        if confidence is None:
            raise ResponseParsingError(
                "Failed to determine confidence:\n " + self.transcript()
            )
        result = float(confidence.group(1)) / 100
        if not (0 <= result <= 1):
            raise ResponseParsingError(
                f"Failed to determine confidence, got invalid value {result}:\n {bot.transcript()}"
            )
        return result

    def structure(self, target: Type[T]) -> T:
        """Takes the last message from this chatbot and tries to convert it
        to a structured value of type T."""

        # Please note! Although this method presents a nice clean high level API,
        # it is in fact a collection of unprincipled hacks and workarounds.
        #
        # When adding a new evaluation or testing a new model you are *extremely
        # strongly encouraged* to add new tests to test_response_structuring.py
        # to make sure that it correctly structures the responses you are seeing.

        structuring_bot = self.clone()

        adapter = TypeAdapter(target)

        checked_for_answer_already = False
        if any(refusal in self.last_response.lower() for refusal in REFUSAL_PHRASES):
            checked_for_answer_already = True
            if not self.did_the_bot_answer():
                raise FailedToAnswer(
                    "Chatbot failed to provide an answer to the question."
                )
        response = structuring_bot.chat(
            STRUCTURING_PROMPT.format(SCHEMA=json.dumps(adapter.json_schema()))
        )

        for parsed in reversed(extract_json_objects(response)):
            try:
                return conform_json_to_type(target, parsed)
            except ValidationError:
                pass

        if checked_for_answer_already or self.did_the_bot_answer():
            raise ResponseParsingError(
                f"Something went wrong in interacting with chatbot:\n\n{structuring_bot.transcript()}"
            )
        else:
            raise FailedToAnswer("Chatbot failed to provide an answer to the question.")

    def new_client(self) -> Any: ...

    def complete(self) -> str:
        """Complete the current conversation with the assistant."""
        ...


class DummyChatbot(Chatbot):
    def new_client(self) -> Any:
        return None

    def complete(self) -> str:
        raise TypeError("Dummy chatbot cannot complete conversations.")
