from typing import TypedDict, Literal, Any, Type, TypeVar, cast
from pydantic import TypeAdapter, ValidationError
import json
import re
import os
from contextlib import contextmanager
import sqlite3
from typing import Iterator, Callable
from functools import lru_cache
from pydantic import BaseModel
from threading import RLock


T = TypeVar("T")
Role = Literal["user", "assistant", "system"]


class Message(TypedDict):
    role: Role
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
Please provide a structured representation of your previous answer as a JSON value matching the following schema: {SCHEMA}.

Important instructions:
- Your answer is some value matching the schema, not the schema itself.
- Carefully read the question and consider the specific information it is asking for.
- Include only the direct answer to the question, without any additional explanations, qualifications, or extraneous text.
- Don't wrap the answer in any unnecessary JSON objects. If the expected output is a list, provide only the list. If it's a single value, provide only that value.
- Each string element should be concise and not include any detailed explanations.
- If your previous answer repeats any part of the question, do not include that in your structured answer. e.g. if your answer to "What is the first month of the year?" is "The first month of the year is January", only "January" should appear in the structured answer.
- Make sure not to miss out any parts from your answer, even if it doesn't look like a correct answer to the question. e.g. if your answer contains a list, be sure to include every element, even if some of the elements look wrong.
- Don't attempt to correct or modify your previous answer. Even if you now realize it was incorrect or incomplete, provide a structured representation of the original answer.

""".strip()

LISTING_PROMPT = """
Please provide a markdown style numbered list that contains just your previous answers.

Important instructions:
* Read the question carefully, determine what an answer to it looks like, and only include those answers in your list. For example, if I asked for a list of numbers, each list item should contain only a number and no additional text.
* List items should not contain any bracketed asides about the answer unless they are directly a part of the requested answer.
* Omit all clarifying text, context, caveats, notes or explanations.
* Do not include any explanations of why those answers are the right ones.
* Do not in any way modify your previous answer, even if you now realise it's incorrect.
""".strip()

EXTRACT_ANSWER_PROMPT = """
Please concisely state just the part of your previous response that contains the answer as a JSON formatted string.

Important instructions:
    * Do not include any unnecessary punctuation in your answer.
    * Do not provide any additional context or information.
    * Do not restate any part of the question in your answer or explanations.
    * Do not contain any explanations of why it's the answer.
"""


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

    if target_type == str and isinstance(json_object, (int, float)):
        return str(json_object)  # type: ignore
    if isinstance(json_object, dict):
        if len(json_object) == 1:
            (wrapped,) = json_object.values()
            return conform_json_to_type(target_type, wrapped)
        elif len(json_object) == 2 and "type" in json_object:
            (wrapped,) = [v for k, v in json_object.items() if k != "type"]
            return conform_json_to_type(target_type, wrapped)
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


YES_NO_ANSWER = re.compile(r"^(yes|no)\b", re.IGNORECASE)

NUMERIC_LIST = re.compile(r"^[0-9]+\. (.+)$")


def messages_to_transcript(messages):
    parts = []
    for message in messages:
        parts.append(f"{message['role']}: {message['content']}\n")
    return "\n".join(parts)


class Chatbot:
    @classmethod
    def subclass_for_model(cls, model):
        if model == "dummy":
            return DummyChatbot
        elif model.startswith("claude"):
            from foundationevals.chatbots.claude import Claude

            return Claude
        elif model.startswith("gpt"):
            from foundationevals.chatbots.gpt import GPT

            return GPT
        else:
            from foundationevals.chatbots.ollama import Ollama

            return Ollama

    def __new__(cls, model, **kwargs):
        kwargs["model"] = model
        subclass = cls.subclass_for_model(model)
        result = object.__new__(subclass)  # type: ignore
        result.__init__(**kwargs)
        return result

    def __init__(
        self,
        model: str,
        cache: "ChatCache | str | None" = None,
        index: int | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        messages: list[Message] | None = None,
    ):
        self.model: str = model
        self.messages: list[Message] = [] if messages is None else list(messages)
        self.temperature: float = temperature
        self.max_tokens = max_tokens
        if cache is None:
            cache = ChatCache.default_cache()
        elif isinstance(cache, str):
            cache = ChatCache(cache)
        assert isinstance(cache, ChatCache)
        self.cache = cache
        self.index = index
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
        if response is not None:
            result = response
        elif self.index is not None:
            result_id = self.cache.completion(
                self.cache.messages_to_transcript_id(self.messages),
                self.model,
                self.index,
                self.complete,
            )
            saved_message = self.cache.get_message(result_id)
            assert saved_message is not None
            result = saved_message.content
        else:
            result = self.complete()
        self.messages.append({"role": "assistant", "content": result})
        return result

    def clone(self, **kwargs) -> "Chatbot":
        """Create a new chatbot instance with the same state."""
        kwargs.setdefault("model", self.model)
        kwargs.setdefault("temperature", self.temperature)
        kwargs.setdefault("messages", self.messages)
        kwargs.setdefault("index", self.index)
        kwargs.setdefault("cache", self.cache)
        result = self.__class__(**kwargs)
        self.children.append(result)
        return result

    @property
    def client(self):
        if self.__client is None:
            self.__client = self.new_client()
        return self.__client

    def transcript(self) -> str:
        return messages_to_transcript(self.messages)

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
            "Did your response contain the information I requested, "
            "or was there a reason you could not or did not answer it "
            "as asked? Don't apologise, don't correct your answer. "
            "Just tell me if you provided the requested information."
        )

        match = YES_NO_ANSWER.match(check_for_refusal)

        if match is None:
            # If the bot didn't say yes or no and has started behaving
            # in a weaselly LLM nonsense way we assume this was a refusal
            # and the dumb check is more likely to be useful than its
            # follow up where it tries to hallucinate its way into
            # deciding that it's a good Bing after all.
            lower_check = check_for_refusal.lower()
            for phrase in REFUSAL_PHRASES:
                if phrase in lower_check:
                    return False

            check_for_refusal = refusal_checking_bot.chat(
                'Please just say "yes" if it provided the requested information or "no" if it did not. Don\'t apologise or explain.'
            )
            match = YES_NO_ANSWER.match(check_for_refusal)

        if match is None:
            raise ResponseParsingError(
                "Could not determine whether chatbot answered the question:\n\n"
                + refusal_checking_bot.transcript()
            )

        answer = match.group(0).lower()
        assert answer in ("yes", "no")
        if answer == "yes":
            # ok but did it really?
            refusal_checking_bot.chat(
                "Please point to the exact words in your response that answered my question."
            )
            confirmation = refusal_checking_bot.chat(
                "Given this, do you still think your response contained the information I requested?"
            )
            match = YES_NO_ANSWER.match(confirmation)
            if match is None:
                raise ResponseParsingError(
                    "Could not determine whether chatbot answered the question:\n\n"
                    + refusal_checking_bot.transcript()
                )
            answer = match.group(0).lower()
            assert answer in ("yes", "no")
            return answer == "yes"
        else:
            return False

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
        checked_for_answer_already = False
        if any(refusal in self.last_response.lower() for refusal in REFUSAL_PHRASES):
            checked_for_answer_already = True
            if not self.did_the_bot_answer():
                raise FailedToAnswer(
                    "Chatbot failed to provide an answer to the question."
                )

        if target == bool:
            match = YES_NO_ANSWER.match(self.last_response)

            if match is not None:
                answer = match.group(0).lower()

                if answer == "yes":
                    return True  # type: ignore
                else:
                    assert answer == "no"
                    return False  # type: ignore

        if target == list[str]:
            response = self.clone().chat(LISTING_PROMPT)

            result = []

            for line in response.splitlines():
                match = NUMERIC_LIST.match(line)
                if match is not None:
                    result.append(match.group(1))

            if result:
                return result  # type: ignore

        structuring_bot = self.clone()

        adapter = TypeAdapter(target)

        if target == str:
            response = structuring_bot.chat(EXTRACT_ANSWER_PROMPT)
        else:
            response = structuring_bot.chat(
                STRUCTURING_PROMPT.format(
                    SCHEMA=json.dumps(adapter.json_schema()),
                ),
            )

        validation_error = False
        for parsed in reversed(extract_json_objects(response)):
            try:
                return conform_json_to_type(target, parsed)
            except ValidationError:
                validation_error = True

        if validation_error:
            response = structuring_bot.chat(
                f"That didn't match the schema {json.dumps(adapter.json_schema())}. Please try again."
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
        return os.urandom(16).hex()


CREATE_TRANSCRIPT_SQL = """
create table if not exists transcripts(
    id integer primary key,
    parent integer not null,
    role text not null,
    content text not null,
    unique (parent, role, content)
)
"""

CREATE_COMPLETIONS_SQL = """
create table if not exists completions(
    parent integer not null,
    model text not null,
    ix integer not null,
    next_message integer not null,
    unique (parent, model, ix),
    foreign key (next_message) references transcripts(id)
)
"""


class SavedMessage(BaseModel):
    id: int
    parent: int
    role: Role
    content: str


class ChatCache:
    CONNECTIONS = {}

    @classmethod
    def default_cache(cls):
        try:
            return cls.__default_cache
        except AttributeError:
            pass
        cls.__default_cache = ChatCache(os.environ.get("CHAT_CACHE_DB", ":memory:"))
        return cls.__default_cache

    def __init__(self, connection):
        if isinstance(connection, str):
            try:
                connection = self.CONNECTIONS[connection]
            except KeyError:
                db = sqlite3.connect(connection)
                self.CONNECTIONS[connection] = db
                connection = db

        self.__connection = connection
        self.__lock = RLock()

        with self.__cursor() as c:
            c.execute(CREATE_TRANSCRIPT_SQL)
            c.execute(CREATE_COMPLETIONS_SQL)

        self.get_message = lru_cache(1024)(self.get_message)
        self.__next_transcript = lru_cache(1024)(self.__next_transcript)

    @contextmanager
    def __cursor(self) -> Iterator[sqlite3.Cursor]:
        with self.__lock:
            with self.__connection:
                cursor = self.__connection.cursor()
                try:
                    yield cursor
                finally:
                    cursor.close()

    def messages_to_transcript_id(self, transcript: list[Message]) -> int:
        result = 0
        for message in transcript:
            result = self.__next_transcript(parent=result, **message)
        return result

    def get_message(self, id: int) -> SavedMessage | None:
        if id == 0:
            return None
        with self.__cursor() as cursor:
            cursor.execute(
                "select parent, role, content from transcripts where id = ?",
                (id,),
            )
            parent, role, content = cursor.fetchone()
            assert parent < id
            return SavedMessage(id=id, parent=parent, role=role, content=content)

    def id_to_messages(self, transcript_id: int) -> list[Message]:
        result = []
        while True:
            message = self.get_message(transcript_id)
            if message is None:
                break
            result.append(
                {
                    "role": message.role,
                    "content": message.content,
                }
            )
            transcript_id = message.parent
        result.reverse()
        return result

    def completion(
        self, transcript_id: int, model: str, index: int, complete: Callable[[], str]
    ) -> int:
        with self.__cursor() as cursor:
            cursor.execute(
                "select next_message from completions where parent = ? and model = ? and ix = ?",
                (
                    transcript_id,
                    model,
                    index,
                ),
            )
            for (next_message,) in cursor:
                return next_message

        result = self.__next_transcript(
            parent=transcript_id, content=complete(), role="assistant"
        )

        with self.__cursor() as cursor:
            cursor.execute(
                "insert into completions(parent, model, ix, next_message) values(?, ?, ?, ?) on conflict do nothing",
                (transcript_id, model, index, result),
            )
            cursor.execute(
                "select next_message from completions where parent = ? and model = ? and ix = ?",
                (
                    transcript_id,
                    model,
                    index,
                ),
            )
            for (result,) in cursor:
                return result
            assert False, "Unreachable"

    def __insert_or_get(self, table: str, **kwargs) -> int:
        columns = []
        values = []
        for k, v in kwargs.items():
            columns.append(k)
            values.append(v)

        clause = " and ".join([f"{column} = ?" for column in columns])
        select_statement = f"select id from {table} where {clause}"

        with self.__cursor() as cursor:
            cursor.execute(select_statement, tuple(values))
            for (id,) in cursor:
                return id
            cursor.execute(
                f"insert into transcripts({', '.join(columns)}) values({', '.join(['?'] * len(columns))}) returning id",
                tuple(values),
            )
            for (id,) in cursor:
                return id
            assert False, "Unreachable"

    def __next_transcript(self, parent: int | None, role: Role, content: str) -> int:
        return self.__insert_or_get(
            "transcripts", parent=parent or 0, role=role, content=content
        )
