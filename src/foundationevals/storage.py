import os
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import lru_cache
from threading import RLock
from typing import Literal, TypedDict

from pydantic import BaseModel

Role = Literal["user", "assistant", "system"]


class Message(TypedDict):
    role: Role
    content: str


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
    temperature float not null,
    ix integer not null,
    next_message integer not null,
    unique (parent, model, temperature, ix),
    foreign key (next_message) references transcripts(id)
)
"""


class SavedMessage(BaseModel):
    id: int
    parent: int
    role: Role
    content: str


class Storage:
    CONNECTIONS = {}

    @classmethod
    def default_storage(cls):
        try:
            return cls.__default_cache
        except AttributeError:
            pass
        cls.__default_cache = Storage(os.environ.get("CHAT_CACHE_DB", ":memory:"))
        return cls.__default_cache

    def __init__(self, connection):
        if isinstance(connection, str):
            try:
                connection = self.CONNECTIONS[connection]
            except KeyError:
                db = sqlite3.connect(connection)
                self.CONNECTIONS[connection] = db
                connection = db

        self.connection = connection
        self.__lock = RLock()

        with self.cursor() as c:
            c.execute(CREATE_TRANSCRIPT_SQL)
            c.execute(CREATE_COMPLETIONS_SQL)

        self.get_message = lru_cache(1024)(self.get_message)
        self.__next_transcript = lru_cache(1024)(self.__next_transcript)

    @contextmanager
    def cursor(self) -> Iterator[sqlite3.Cursor]:
        with self.__lock, self.connection:
            cursor = self.connection.cursor()
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
        with self.cursor() as cursor:
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
        self,
        transcript_id: int,
        model: str,
        temperature: float,
        index: int,
        complete: Callable[[], str],
    ) -> int:
        with self.cursor() as cursor:
            cursor.execute(
                "select next_message from completions where parent = ? and model = ? and temperature = ? and ix = ?",
                (
                    transcript_id,
                    model,
                    temperature,
                    index,
                ),
            )
            for (next_message,) in cursor:
                return next_message

        result = self.__next_transcript(
            parent=transcript_id, content=complete(), role="assistant"
        )

        with self.cursor() as cursor:
            cursor.execute(
                "insert into completions(parent, model,temperature,  ix, next_message) values(?, ?, ?, ?, ?) on conflict do nothing",
                (transcript_id, model, temperature, index, result),
            )
            cursor.execute(
                "select next_message from completions where parent = ? and model = ? and temperature = ? and ix = ?",
                (
                    transcript_id,
                    model,
                    temperature,
                    index,
                ),
            )
            for (result,) in cursor:
                return result
            raise AssertionError("Unreachable")

    def __insert_or_get(self, table: str, **kwargs) -> int:
        columns = []
        values = []
        for k, v in kwargs.items():
            columns.append(k)
            values.append(v)

        clause = " and ".join([f"{column} = ?" for column in columns])
        select_statement = f"select id from {table} where {clause}"

        with self.cursor() as cursor:
            cursor.execute(select_statement, tuple(values))
            for (id,) in cursor:
                return id
            cursor.execute(
                f"insert into transcripts({', '.join(columns)}) values({', '.join(['?'] * len(columns))}) returning id",
                tuple(values),
            )
            for (id,) in cursor:
                return id
            raise AssertionError("Unreachable")

    def __next_transcript(self, parent: int | None, role: Role, content: str) -> int:
        return self.__insert_or_get(
            "transcripts", parent=parent or 0, role=role, content=content
        )
