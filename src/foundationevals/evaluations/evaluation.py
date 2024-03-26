from random import Random
from typing import Generic, TypeVar, Type, Callable
from enum import Enum
from foundationevals.chatbots.base import Chatbot, FailedToAnswer, Message
from pydantic import TypeAdapter
from abc import ABC, abstractmethod
from pydantic import BaseModel, ConfigDict
from concurrent.futures import ThreadPoolExecutor


Problem = TypeVar("Problem")
Answer = TypeVar("Answer")


class EvaluationResultStatus(Enum):
    INCOMPLETE = 0
    NONANSWER = 1
    ANSWER = 2


class NoValidAnswer(Exception):
    pass


class ReportedStatus(Enum):
    CORRECT = 0
    INCORRECT = 1
    NONANSWER = 2


class SingleProblemReport(BaseModel, Generic[Problem, Answer]):
    problem: Problem
    answer: Answer | None
    confidence: float | None
    status: ReportedStatus
    errors: set[str]
    notes: list[str]
    transcripts: list[list[Message]]


class SingleProblemEvaluation(Generic[Problem, Answer]):
    def __init__(self, chatbot: Chatbot, problem: Problem, answer_type: Type[Answer]):
        assert answer_type is not None
        self.chatbot = chatbot
        self.problem: Problem = problem
        self.status: EvaluationResultStatus = EvaluationResultStatus.INCOMPLETE
        self.errors: set[str] = set()
        self.notes: list[str] = []
        self.answer: Answer | None = None
        self.answer_type: Type[Answer] = answer_type
        self.confidence: float | None = None

    def mark_nonanswer(self):
        self.__check_incomplete()
        self.status = EvaluationResultStatus.NONANSWER

    def mark_answer(self, answer: Answer):
        self.__check_incomplete()
        self.answer = answer
        self.status = EvaluationResultStatus.ANSWER

    def record_confidence(self, confidence: float):
        if not (0 <= confidence <= 1):
            raise ValueError(f"Invalid confidence value {confidence}.")
        if self.confidence is not None:
            raise ValueError("Cannot note confidence more than once.")
        self.confidence = confidence

    def add_error(self, error: str):
        self.errors.add(error)

    def add_note(self, note: str):
        self.notes.append(note)

    def __check_incomplete(self):
        if self.status != EvaluationResultStatus.INCOMPLETE:
            raise ValueError("Cannot change the status of a completed scorecard.")

    def parse(self) -> Answer:
        """Attempts to parse the chatbot's last response into a structured answer
        of the right type, setting all of the main scorecard attributes.

        This is a convenience method. Some evaluations may need to mark answers
        on the score card more directly if `chatbot.structure()` cannot be relied
        on to do the right thing.
        """
        self.__check_incomplete()
        try:
            result = self.chatbot.structure(self.answer_type)
            self.mark_answer(result)
            self.record_confidence(self.chatbot.confidence())
            self.chatbot.freeze()
            return result
        except FailedToAnswer:
            self.mark_nonanswer()
            raise NoValidAnswer()

    def report(self) -> SingleProblemReport[Problem, Answer]:
        if self.status == EvaluationResultStatus.INCOMPLETE:
            raise ValueError("Cannot report an incomplete scorecard.")

        if self.confidence is None and self.status != EvaluationResultStatus.NONANSWER:
            raise ValueError("Confidence must be recorded before reporting.")

        if self.status == EvaluationResultStatus.NONANSWER:
            status = ReportedStatus.NONANSWER
        elif self.errors:
            status = ReportedStatus.INCORRECT
        else:
            status = ReportedStatus.CORRECT

        transcripts = []
        stack = [self.chatbot]
        while stack:
            chatbot = stack.pop()
            transcripts.append(list(chatbot.messages))
            stack.extend(chatbot.children)

        return SingleProblemReport(
            problem=self.problem,
            answer=self.answer,
            confidence=self.confidence,
            status=status,
            errors=set(self.errors),
            notes=list(self.notes),
            transcripts=transcripts,
        )


class ProblemSet(ABC, Generic[Problem]):
    def __init__(self, problem_type: Type[Problem] | None = None):
        self.__problem_type = problem_type
        self.__adapter = None

    @property
    def problem_type(self) -> Type[Problem]:
        if self.__problem_type is None:
            for cls in self.__class__.mro():
                if hasattr(cls, "__orig_bases__"):
                    for base in cls.__orig_bases__:
                        if base.__origin__ == ProblemSet:
                            problem_type = base.__args__[0]
                            if isinstance(problem_type, type):
                                self.__problem_type = problem_type
                                break
            else:
                raise ValueError(
                    "Could not determine problem type. Please set it explicitly."
                )
        assert self.__problem_type is not None
        return self.__problem_type  # type: ignore

    @property
    def type_adapter(self) -> TypeAdapter[Problem]:
        if self.__adapter is None:
            self.__adapter = TypeAdapter(self.problem_type)
        return self.__adapter

    def dump(self, problem: Problem) -> bytes:
        return self.type_adapter.dump_json(problem)

    def load(self, data: bytes) -> Problem:
        return self.type_adapter.validate_json(data)

    def reduce(
        self, problem: Problem, is_interesting: Callable[[Problem], bool]
    ) -> Problem:
        return problem

    @abstractmethod
    def generate(self, random: Random) -> Problem: ...


def cached_property(fn):
    attr_name = f"_{fn.__name__}"

    @property
    def wrapper(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)

    return wrapper


class FullReport(BaseModel, Generic[Problem, Answer]):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: str

    initial_random_sample: list[SingleProblemReport[Problem, Answer]]
    other_samples: list[SingleProblemReport[Problem, Answer]]
    reduced_exemplar: SingleProblemEvaluation[Problem, Answer] | None

    @property
    def sample_size(self):
        return len(self.initial_random_sample)

    @cached_property
    def answered_questions(self):
        return len(
            [
                report
                for report in self.initial_random_sample
                if report.status != ReportedStatus.NONANSWER
            ]
        )

    @cached_property
    def correct_answers(self):
        return len(
            [
                report
                for report in self.initial_random_sample
                if report.status == ReportedStatus.CORRECT
            ]
        )

    @property
    def answer_rate(self):
        return self.answered_questions / self.sample_size

    @property
    def correct_answer_rate(self):
        if self.answered_questions > 0:
            return self.correct_answers / self.answered_questions
        else:
            return float("nan")

    @cached_property
    def confidence_calibration(self):
        """Returns a score between 0 and 1 that measures how well
        calibrated the chatbot's confidence is. This is 1 - the Brier
        score, so a score of 1 means perfect calibration and a score of
        0 means the worst possible calibration."""
        brier_score_sum = 0.0

        for report in self.initial_random_sample:
            if report.status == ReportedStatus.NONANSWER:
                continue
            confidence = report.confidence

            if report.status == ReportedStatus.CORRECT:
                brier_score_sum += (1 - confidence) ** 2
            else:
                brier_score_sum += confidence**2
        result = 1.0 - (brier_score_sum / self.answered_questions)
        assert 0.0 <= result <= 1.0
        return result


def run_basic_evaluation(
    problem_set: ProblemSet[Problem],
    evaluation: Callable[[SingleProblemEvaluation[Problem, Answer]], None],
    *,
    chatbot: Chatbot,
    random: Random | None = None,
    n_samples=1000,
    parallelism=1,
    reduce=False,
    answer_type=None,
):
    chatbot.freeze()
    if random is None:
        random = Random()

    if answer_type is None:
        annotations = dict(evaluation.__annotations__)
        annotations.pop("return", None)
        if len(annotations) == 1:
            (evaluation_arg,) = annotations.values()
            answer_type = evaluation_arg.__args__[1]

    if answer_type is None:
        raise ValueError(
            "Cannot infer answer type from evaluation function. "
            "Please add a return annotation to the function or pass "
            "answer_type explicitly."
        )

    def evaluate_problem(problem: Problem) -> SingleProblemReport[Problem, Answer]:
        assert answer_type is not None
        problem_evaluation = SingleProblemEvaluation(
            chatbot=chatbot.clone(),
            problem=problem,
            answer_type=answer_type,
        )
        try:
            evaluation(problem_evaluation)
        except NoValidAnswer:
            pass
        return problem_evaluation.report()

    with ThreadPoolExecutor(max_workers=parallelism) as executor:
        initial_results = list(
            executor.map(
                evaluate_problem,
                (problem_set.generate(random) for _ in range(n_samples)),
            )
        )

    assert not reduce

    return FullReport(
        model=chatbot.model,
        initial_random_sample=initial_results,
        other_samples=[],
        reduced_exemplar=None,
    )
