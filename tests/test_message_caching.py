import os
import sqlite3

from hypothesis import given

from foundationevals.chatbots.base import Chatbot, Message
from foundationevals.storage import Storage


@given(...)
def test_can_roundtrip_messages(messages: list[Message]):
    con = sqlite3.connect(":memory:")
    storage = Storage(con)

    id = storage.messages_to_transcript_id(messages)
    assert (id > 0) == (len(messages) > 0)
    round_tripped = storage.id_to_messages(id)
    assert messages == round_tripped


def test_caches_completions_by_index_and_model():
    con = sqlite3.connect(":memory:")
    storage = Storage(con)

    def new_completion() -> str:
        return os.urandom(16).hex()

    runs = []

    for _ in range(2):
        run = []
        runs.append(run)
        for model in ["dummy1", "dummy2"]:
            for index in range(2):
                run.append(storage.completion(0, model, 0.1, index, new_completion))

    for run in runs:
        assert len(set(run)) == len(run)

    x, y = runs
    assert x == y


def test_warm_chatbot_interactions_are_storaged():
    x = Chatbot("dummy", index=0, temperature=0.1)
    y = Chatbot("dummy", index=0, temperature=0.1)

    assert x.chat("Hello world") == y.chat("Hello world")


def test_warm_chatbot_interactions_with_different_index_are_storaged_differently():
    x = Chatbot("dummy", index=0, temperature=0.1)
    y = Chatbot("dummy", index=1, temperature=0.1)

    assert x.chat("Hello world") != y.chat("Hello world")


def test_chatbot_interactions_with_no_index_are_not_storaged():
    x = Chatbot("dummy")
    y = Chatbot("dummy")

    assert x.chat("Hello world") != y.chat("Hello world")


def test_caches_across_cloning():
    x = Chatbot("dummy", index=0)
    y = x.clone()
    assert x.chat("hello") == y.chat("hello")


def test_commits_storage_to_disk(tmpdir):
    db = str(tmpdir / "chat.sqlite3")
    conn = sqlite3.connect(db)

    x = Chatbot("dummy", index=0, storage=Storage(conn))
    assert x.storage != Storage.default_storage()

    chat1 = x.chat("hello")
    conn.close()

    y = Chatbot("dummy", index=0, storage=db)
    assert y.storage != Storage.default_storage()
    assert y.chat("hello") == chat1


def test_storage_depends_on_temperature():
    x = Chatbot("dummy", index=0, temperature=0.1)
    y = Chatbot("dummy", index=0, temperature=0.15)
    assert x.chat("hello") != y.chat("hello")


def test_always_caches_low_temperature_to_same_index():
    x = Chatbot("dummy", index=0, temperature=0)
    y = Chatbot("dummy", index=1, temperature=0)
    assert x.chat("hello") == y.chat("hello")
