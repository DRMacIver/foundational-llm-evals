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
