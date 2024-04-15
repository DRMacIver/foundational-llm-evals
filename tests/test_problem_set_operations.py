import pytest
from foundationevals.evaluations.arithmetic import AdditionEvaluation
from foundationevals.evaluations.wordgen import WordConstraintsProblemSet
from hypothesis import given, strategies as st, settings, HealthCheck
from hypothesis.errors import Frozen, StopTest

problem_sets = pytest.mark.parametrize(
    "problem_set",
    [
        WordConstraintsProblemSet(),
        AdditionEvaluation().problem_set,
    ],
)


@problem_sets
def test_has_problem_type(problem_set):
    assert problem_set.problem_type is not None


@problem_sets
@given(rnd=st.randoms(use_true_random=False))
@settings(deadline=None)
def test_can_serialize_arbitrary_problems(problem_set, rnd):
    problem = problem_set.generate(rnd)
    assert problem_set.load(problem_set.dump(problem)) == problem


@problem_sets
@given(rnd=st.randoms(use_true_random=False), frequency=st.integers(1, 10))
@settings(
    deadline=None,
    suppress_health_check=[HealthCheck.large_base_example, HealthCheck.data_too_large],
    max_examples=2,
)
def test_can_reduce(problem_set, rnd, frequency):
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
