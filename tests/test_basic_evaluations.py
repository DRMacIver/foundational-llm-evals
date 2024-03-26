from foundationevals.evaluations.evaluation import (
    SingleProblemEvaluation,
    run_basic_evaluation,
    ProblemSet,
)
from random import Random
from foundationevals.chatbots import Chatbot


class SmallIntegerProblemSet(ProblemSet[int]):
    def generate(self, random: Random) -> int:
        return random.randint(1, 100)


def is_this_big(evaluation: SingleProblemEvaluation[int, bool]) -> None:
    assert evaluation.answer_type == bool
    evaluation.chatbot.chat(f"Is {evaluation.problem} bigger than 10?")
    answer = evaluation.parse()
    assert isinstance(answer, bool)
    correct_answer = evaluation.problem > 10

    evaluation.add_note(f"answer: {answer}, correct answer: {correct_answer}")

    if answer != correct_answer:
        if answer:
            evaluation.add_error("Incorrectly reported the number as big.")
        else:
            evaluation.add_error("Incorrectly reported the number as small.")


def test_bigness_evaluation():
    report = run_basic_evaluation(
        SmallIntegerProblemSet(),
        is_this_big,
        # Normally we'd run this with llama2 because it's cheaper, but
        # llama2 gets this problem reliably wrong??
        chatbot=Chatbot("mistral"),
        n_samples=10,
    )
    assert report.correct_answer_rate > 0
    assert report.answer_rate > 0
    assert 0 < report.confidence_calibration <= 1
