from hypothesis import given, strategies as st, example

from foundationevals.chatbots.base import Message, ChatCache, Chatbot
import sqlite3
import os


@given(...)
def test_can_roundtrip_messages(messages: list[Message]):
    con = sqlite3.connect(":memory:")
    cache = ChatCache(con)

    id = cache.messages_to_transcript_id(messages)
    assert (id > 0) == (len(messages) > 0)
    round_tripped = cache.id_to_messages(id)
    assert messages == round_tripped


def test_caches_completions_by_index_and_model():
    con = sqlite3.connect(":memory:")
    cache = ChatCache(con)

    def new_completion() -> str:
        return os.urandom(16).hex()

    runs = []

    for _ in range(2):
        run = []
        runs.append(run)
        for model in ["dummy1", "dummy2"]:
            for index in range(2):
                run.append(cache.completion(0, model, index, new_completion))

    for run in runs:
        assert len(set(run)) == len(run)

    x, y = runs
    assert x == y


def test_chatbot_interactions_are_cached():
    x = Chatbot("dummy", index=0)
    y = Chatbot("dummy", index=0)

    assert x.chat("Hello world") == y.chat("Hello world")


def test_chatbot_interactions_with_different_index_are_cached_differently():
    x = Chatbot("dummy", index=0)
    y = Chatbot("dummy", index=1)

    assert x.chat("Hello world") != y.chat("Hello world")


def test_chatbot_interactions_with_no_index_are_not_cached():
    x = Chatbot("dummy")
    y = Chatbot("dummy")

    assert x.chat("Hello world") != y.chat("Hello world")


def test_caches_across_cloning():
    x = Chatbot("dummy", index=0)
    y = x.clone()
    assert x.chat("hello") == y.chat("hello")


def test_commits_cache_to_disk(tmpdir):
    db = str(tmpdir / "chat.sqlite3")
    conn = sqlite3.connect(db)

    x = Chatbot("dummy", index=0, cache=ChatCache(conn))
    assert x.cache != ChatCache.default_cache()

    chat1 = x.chat("hello")
    conn.close()

    y = Chatbot("dummy", index=0, cache=db)
    assert y.cache != ChatCache.default_cache()
    assert y.chat("hello") == chat1
