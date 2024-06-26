import os
import sqlite3
from random import Random
from typing import Any, TypeVar

import pytest

from foundationevals.chatbots import Chatbot
from foundationevals.evaluations.arithmetic import (
    AdditionEvaluation,
    BaseConversionEvaluation,
)
from foundationevals.evaluations.evaluation import (
    BasicEvaluation,
    SingleProblemEvaluation,
)
from foundationevals.evaluations.wordgen import (
    WordGenerationEvaluation,
)
from foundationevals.storage import Storage


def in_memory_storage():
    return Storage(sqlite3.connect(":memory:"))


class TrivialEvaluation(BasicEvaluation[int]):
    def generate(self, random: Random) -> int:
        return random.randint(0, 1000)

    def run_single_evaluation(self, evaluation: SingleProblemEvaluation[int]) -> None:
        m = evaluation.problem
        evaluation.chatbot.chat(f"Is {m} bigger than 10?")
        answer = evaluation.parse(bool)
        if answer != (m > 10):
            evaluation.add_error("Wrong answer")
        elif m > 100:
            evaluation.add_error("Fake error for testing purposes")


CACHE_ONLY = os.environ.get("UPDATE_CACHE", "") != "true"

evaluations = pytest.mark.parametrize(
    "evaluation",
    [
        TrivialEvaluation,
        WordGenerationEvaluation,
        AdditionEvaluation,
        BaseConversionEvaluation,
    ],
)


@evaluations
def test_has_concrete_problem_types(evaluation):
    typ = evaluation().problem_set.problem_type
    assert not isinstance(typ, TypeVar)


@evaluations
def test_can_reduce_generated_problems(evaluation):
    ev = evaluation()
    ps = ev.problem_set
    initial = ps.generate(Random(0))
    reduced = ps.reduce(initial, lambda x: True)
    assert ps.reduction_key(reduced) <= ps.reduction_key(initial)


@evaluations
def test_problem_generation_is_stable(evaluation):
    ev = evaluation()
    ps = ev.problem_set
    assert ps.generate(Random(13)) == ps.generate(Random(13))


@evaluations
def test_dummy_has_a_zero_answer_rate(evaluation):
    ev = evaluation()

    report = ev.run(
        chatbot=Chatbot("dummy"),
        n_samples=10,
        reduce=True,
        storage=in_memory_storage(),
    )
    assert report.answer_rate == 0
    assert report.reduced_exemplar is None


@evaluations
def test_can_produce_answers(model, evaluation: type[BasicEvaluation[Any]]):
    ev = evaluation()
    cache_file = os.path.join(os.path.dirname(__file__), "cache.sqlite3")

    report = ev.run(
        chatbot=Chatbot(
            model, temperature=0, index=0, storage=cache_file, cache_only=CACHE_ONLY
        ),
        n_samples=5,
        reduce=False,
        stop_on_first_failure=False,
        random=Random(0),
        pb=False,
        reraise_errors=True,
        storage=in_memory_storage(),
    )
    assert report.answer_rate > 0
    assert not report.unhandled_errors


def test_can_reduce_trivial_evaluation(model):
    ev = TrivialEvaluation()
    cache_file = os.path.join(os.path.dirname(__file__), "cache.sqlite3")

    report = ev.run(
        chatbot=Chatbot(
            model, temperature=0, index=0, storage=cache_file, cache_only=CACHE_ONLY
        ),
        n_samples=100,
        reduce=True,
        stop_on_first_failure=True,
        random=Random(0),
        reraise_errors=True,
        pb=False,
        storage=in_memory_storage(),
    )
    assert report.answer_rate > 0
    if report.correct_answer_rate < 1:
        assert report.reduced_exemplar is not None
        reduced = report.reduced_exemplar.problem
        ps = ev.problem_set
        for x in report.initial_random_sample:
            if x.errors:
                assert ps.reduction_key(x.problem) >= ps.reduction_key(reduced)


@evaluations
def test_writes_results_to_database(evaluation):
    ev = evaluation()
    storage = in_memory_storage()

    report1 = ev.run(
        chatbot=Chatbot("dummy"),
        n_samples=10,
        reduce=True,
        storage=storage,
        random=Random(0),
    )

    with storage.cursor() as cursor:
        cursor.execute("select count(*) from single_evaluations")
        (n,) = cursor.fetchone()
        assert n == 10

    report2 = ev.run(
        chatbot=Chatbot("dummy"),
        n_samples=10,
        reduce=True,
        storage=storage,
        random=Random(0),
    )

    assert [s.problem for s in report1.initial_random_sample] == [
        s.problem for s in report2.initial_random_sample
    ]

    assert report1 == report2

    with storage.cursor() as cursor:
        cursor.execute("select count(*) from single_evaluations")
        (n,) = cursor.fetchone()
        assert n == 10
