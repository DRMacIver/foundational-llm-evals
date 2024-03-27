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


def is_this_big(evaluation: SingleProblemEvaluation[int]) -> None:
    evaluation.chatbot.chat(f"Is {evaluation.problem} bigger than 10?")
    answer = evaluation.parse(bool)
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


def always_fails(evaluation: SingleProblemEvaluation[int]) -> None:
    evaluation.chatbot.chat(f"Is {evaluation.problem} bigger than 10?", response="Yes")
    evaluation.mark_answered()
    evaluation.record_confidence(1.0)
    evaluation.add_error("This evaluation should always fail.")


def test_always_failing_example():
    report = run_basic_evaluation(
        SmallIntegerProblemSet(),
        always_fails,
        chatbot=Chatbot("dummy"),
        n_samples=10,
        reduce=True,
    )
    assert report.correct_answer_rate == 0
    assert report.answer_rate == 1

    assert report.reduced_exemplar is not None
    assert report.reduced_exemplar.problem == 0


def always_succeeds(evaluation: SingleProblemEvaluation[int]) -> None:
    evaluation.chatbot.chat(
        f"Is {evaluation.problem} bigger than 10?",
        response="Yes" if evaluation.problem > 10 else "No",
    )
    evaluation.mark_answered()
    evaluation.record_confidence(1.0)


def test_always_succeeding_example():
    report = run_basic_evaluation(
        SmallIntegerProblemSet(),
        always_succeeds,
        chatbot=Chatbot("dummy"),
        n_samples=10,
        reduce=True,
    )
    assert report.correct_answer_rate == 1
    assert report.answer_rate == 1

    assert report.reduced_exemplar is None
