import pytest
from foundationevals.evaluations.wordgen import WordConstraintsProblemSet
from hypothesis import given, strategies as st, settings, HealthCheck
from hypothesis.errors import Frozen, StopTest

problem_sets = pytest.mark.parametrize(
    "problem_set_cls",
    [
        WordConstraintsProblemSet,
    ],
)


@problem_sets
@given(rnd=st.randoms(use_true_random=False))
@settings(deadline=None)
def test_can_serialize_arbitrary_problems(problem_set_cls, rnd):
    problem_set = problem_set_cls()
    problem = problem_set.generate(rnd)
    assert problem_set.load(problem_set.dump(problem)) == problem


@problem_sets
@given(rnd=st.randoms(use_true_random=False), frequency=st.integers(1, 10))
@settings(
    deadline=None,
    suppress_health_check=[HealthCheck.large_base_example, HealthCheck.data_too_large],
    max_examples=2,
)
def test_can_reduce(problem_set_cls, rnd, frequency):
    problem_set = problem_set_cls()

    cache = {}
    counter = 0

    def is_interesting(problem):
        nonlocal counter
        key = problem_set.dump(problem)
        try:
            return cache[key]
        except KeyError:
            counter += 1
            result = counter % frequency == 0
            cache[key] = result
            return result

    problem = problem_set.generate(rnd)
    cache[problem_set.dump(problem)] = True

    try:
        problem_set.reduce(
            problem,
            is_interesting,
            parallelism=1,
            random=rnd,
        )
    except* (Frozen, StopTest):
        pass
