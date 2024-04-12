from foundationevals.evaluations.evaluation import ProblemSet, SingleProblemEvaluation
from random import Random


class PairOfPositiveIntegers(ProblemSet[tuple[int, int]]):
    def generate(self, random: Random) -> tuple[int, int]:
        return (
            self.__generate_single_digit(random),
            self.__generate_single_digit(random),
        )

    def __generate_single_digit(self, random):
        n = random.randint(1, 50)
        return random.randrange(0, 10**n)


def can_add_numbers_together(evaluation: SingleProblemEvaluation[tuple[int, int]]):
    m, n = evaluation.problem
    evaluation.chatbot.chat(f"What is {m} + {n}?")
    result = evaluation.parse(int)
    if m + n != result:
        evaluation.add_error("Incorrect sum")


from foundationevals.evaluations.evaluation import (
    ProblemSet,
    SingleProblemEvaluation,
    EvaluationResultStatus,
)
import pydantic


class BaseConversion(pydantic.BaseModel):
    from_base: int = pydantic.Field(ge=2, le=16)
    to_base: int = pydantic.Field(ge=2, le=16)
    integer: int = pydantic.Field(ge=0)

    @pydantic.model_validator(mode="after")
    def bases_are_distinct(self):
        if self.from_base == self.to_base:
            raise ValueError("Bases must be distinct")
        return self


def int_to_base(n, base):
    if n == 0:
        return "0"
    n_orig = n
    parts = []
    assert n >= 0
    assert 2 <= base <= 16

    while n > 0:
        n, digit = divmod(n, base)
        parts.append(f"{digit:x}")
    parts.reverse()
    result = "".join(parts)
    assert int(result, base) == n_orig
    return result


class BaseConversionProblems(ProblemSet[BaseConversion]):
    def generate(self, random):
        integer = random.randrange(0, 10 ** (random.randrange(1, 21)))
        from_base, to_base = random.sample(range(2, 17), 2)
        return BaseConversion(integer=integer, from_base=from_base, to_base=to_base)


def can_convert_between_bases(evaluation: SingleProblemEvaluation[BaseConversion]):
    p = evaluation.problem
    source = int_to_base(p.integer, p.from_base)
    evaluation.chatbot.chat(
        f"{source} is an integer represented in base {p.from_base}. What is it when represented in base {p.to_base}?"
    )
    result = evaluation.parse(str)
    if evaluation.status != EvaluationResultStatus.ANSWER:
        return
    assert isinstance(result, str)

    try:
        answer = int(result, p.to_base)
    except ValueError as e:
        evaluation.add_error("Non-numeric output")
        evaluation.add_note(e.args[0])
    else:
        if answer != p.integer:
            evaluation.add_error("Incorrect conversion")
