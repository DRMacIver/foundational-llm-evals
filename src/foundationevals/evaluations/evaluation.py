import json
import re
import traceback
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from enum import Enum, auto
from random import Random
from typing import Any, Generic, TypeVar

import trio
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError
from tqdm import tqdm

from foundationevals.chatbots.base import Chatbot, FailedToAnswer, Message
from shrinkray.problem import BasicReductionProblem
from shrinkray.reducer import ShrinkRay
from shrinkray.work import Volume, WorkContext

Problem = TypeVar("Problem")
Answer = TypeVar("Answer")


class EvaluationResultStatus(Enum):
    INCOMPLETE = auto()
    NONANSWER = auto()
    ANSWER = auto()
    UNHANDLED_ERROR = auto()


class NoValidAnswer(Exception):
    pass


class ReportedStatus(Enum):
    CORRECT = auto()
    INCORRECT = auto()
    NONANSWER = auto()
    UNHANDLED_ERROR = auto()


class SingleProblemReport(BaseModel, Generic[Problem]):
    problem: Problem
    confidence: float | None
    status: ReportedStatus
    errors: set[str]
    notes: list[str]
    transcripts: list[list[Message]]
    stack_trace: str | None


class SingleProblemEvaluation(Generic[Problem]):
    def __init__(self, chatbot: Chatbot, problem: Problem, reraise_errors: bool):
        self.chatbot = chatbot
        self.problem: Problem = problem
        self.status: EvaluationResultStatus = EvaluationResultStatus.INCOMPLETE
        self.errors: set[str] = set()
        self.notes: list[str] = []
        self.confidence: float | None = None
        self.stack_trace = None
        self.reraise_errors = reraise_errors

    def __mark_status(self, status):
        self.__check_incomplete()
        self.status = status

    def mark_nonanswer(self):
        self.__mark_status(EvaluationResultStatus.NONANSWER)

    def mark_answered(self):
        self.__mark_status(EvaluationResultStatus.ANSWER)

    def record_confidence(self, confidence: float | None = None):
        if confidence is not None and not (0 <= confidence <= 1):
            raise ValueError(f"Invalid confidence value {confidence}.")
        if self.confidence is not None:
            raise ValueError("Cannot note confidence more than once.")
        if confidence is None:
            confidence = self.chatbot.confidence()
        self.confidence = confidence

    def add_error(self, error: str):
        self.errors.add(error)

    def add_note(self, note: str):
        self.notes.append(note)

    def __check_incomplete(self):
        if self.status != EvaluationResultStatus.INCOMPLETE:
            raise ValueError("Cannot change the status of a completed scorecard.")

    def parse(self, answer_type: type[Answer]) -> Answer:
        """Attempts to parse the chatbot's last response into a structured answer
        of the right type, setting all of the main scorecard attributes.

        This is a convenience method. Some evaluations may need to mark answers
        on the score card more directly if `chatbot.structure()` cannot be relied
        on to do the right thing.
        """
        self.__check_incomplete()
        try:
            result = self.chatbot.structure(answer_type)
            self.record_confidence(self.chatbot.confidence())
            self.mark_answered()
            self.chatbot.freeze()
            self.add_note(f"Parsed answer of type {answer_type} as {result}")
            return result
        except FailedToAnswer:
            self.mark_nonanswer()
            raise NoValidAnswer()
        except Exception:
            if self.reraise_errors:
                raise
            self.__mark_status(EvaluationResultStatus.UNHANDLED_ERROR)
            self.stack_trace = traceback.format_exc()
            raise NoValidAnswer()

    def report(self) -> SingleProblemReport[Problem]:
        if self.status == EvaluationResultStatus.INCOMPLETE:
            raise ValueError("Cannot report an incomplete scorecard.")

        if self.confidence is None and self.status not in (
            EvaluationResultStatus.NONANSWER,
            EvaluationResultStatus.UNHANDLED_ERROR,
        ):
            raise ValueError("Confidence must be recorded before reporting.")

        if self.status == EvaluationResultStatus.NONANSWER:
            status = ReportedStatus.NONANSWER
        elif self.status == EvaluationResultStatus.UNHANDLED_ERROR:
            status = ReportedStatus.UNHANDLED_ERROR
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
            confidence=self.confidence,
            status=status,
            errors=set(self.errors),
            notes=list(self.notes),
            transcripts=transcripts,
            stack_trace=self.stack_trace,
        )


class BadData(Exception):
    pass


def extract_generic_parameter(current, parent):
    queue = deque([current])

    while queue:
        cls = queue.popleft()
        if hasattr(cls, "__orig_bases__"):
            for base in cls.__orig_bases__:
                queue.append(base)
                if hasattr(base, "__origin__") and issubclass(base.__origin__, parent):
                    return base.__args__[0]
        else:
            queue.extend(cls.__bases__)
    raise ValueError(
        "Could not determine parameter of {parent.__class__.__name__} in "
        "{current.__class__.__name__}. Please set it explicitly."
    )


class ProblemSet(ABC, Generic[Problem]):
    def __init__(self, problem_type: type[Problem] | None = None):
        self.__problem_type = problem_type
        self.__adapter = None

    @property
    def problem_type(self) -> type[Problem]:
        if self.__problem_type is None:
            self.__problem_type = extract_generic_parameter(self.__class__, ProblemSet)
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
        try:
            return self.type_adapter.validate_json(data)
        except ValidationError:
            raise BadData()

    def reduction_key(self, problem: Problem) -> Any:
        data = self.dump(problem)
        return (len(data), data)

    def hashable_key(self, problem: Problem) -> Any:
        try:
            hash(problem)
            return problem
        except TypeError:
            pass
        return self.dump(problem)

    def reduce(
        self,
        problem: Problem,
        is_interesting: Callable[[Problem], bool],
        parallelism: int = 1,
        random=None,
    ) -> Problem:
        async def is_interesting_async(data: bytes) -> bool:
            try:
                parsed = self.load(data)
            except BadData:
                await trio.lowlevel.checkpoint()
                return False
            if parallelism <= 1:
                await trio.lowlevel.checkpoint()
                return is_interesting(parsed)
            return await trio.to_thread.run_sync(is_interesting, parsed)

        work = WorkContext(
            random=random or Random(0),
            volume=Volume.quiet,
            parallelism=parallelism,
        )

        reduction_problem = BasicReductionProblem(
            initial=self.dump(problem),
            is_interesting=is_interesting_async,
            work=work,
        )

        reducer = ShrinkRay(reduction_problem)

        trio.run(reducer.run)

        return self.load(reduction_problem.current_test_case)

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


class FullReport(BaseModel, Generic[Problem]):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: str

    initial_random_sample: list[SingleProblemReport[Problem]]
    other_samples: list[SingleProblemReport[Problem]]
    reduced_exemplar: SingleProblemReport[Problem] | None

    @property
    def sample_size(self):
        return len(self.initial_random_sample)

    @cached_property
    def answered_questions(self):
        return len(
            [
                report
                for report in self.initial_random_sample
                if report.status
                in (
                    ReportedStatus.CORRECT,
                    ReportedStatus.INCORRECT,
                )
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

    @cached_property
    def incorrect_answers(self):
        return len(
            [
                report
                for report in self.initial_random_sample
                if report.status == ReportedStatus.INCORRECT
            ]
        )

    @cached_property
    def unhandled_errors(self):
        return [
            report
            for report in (self.initial_random_sample + self.other_samples)
            if report.status == ReportedStatus.UNHANDLED_ERROR
        ]

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
            assert confidence is not None

            if report.status == ReportedStatus.CORRECT:
                brier_score_sum += (1 - confidence) ** 2
            else:
                brier_score_sum += confidence**2
        result = 1.0 - (brier_score_sum / self.answered_questions)
        assert 0.0 <= result <= 1.0
        return result


def camel_to_hyphen(identifier):
    hyphenated = re.sub(r"(?<!^)(?=[A-Z])", "-", identifier)
    hyphenated = hyphenated.lower()
    return hyphenated


class Evaluation(ABC, Generic[Problem]):
    def __init__(self, name=None):
        self.name = name or camel_to_hyphen(self.__class__.__name__)

    @property
    @abstractmethod
    def problem_set(self) -> ProblemSet[Problem]: ...

    @abstractmethod
    def run(
        self,
        *,
        chatbot: Chatbot,
        random: Random | None = None,
        n_samples=1000,
        stop_on_first_failure=False,
        parallelism=1,
        reduce=False,
        pb=True,
        reraise_errors=False,
        storage=None,
    ) -> FullReport[Problem]: ...


class BasicEvaluationProblemSet(ProblemSet[Problem]):
    def __init__(self, evaluation: "BasicEvaluation[Problem]"):
        super().__init__()
        self.evaluation = evaluation
        self.__problem_type = None

    @property
    def problem_type(self):
        if self.__problem_type is None:
            self.__problem_type = extract_generic_parameter(
                self.evaluation.__class__, BasicEvaluation
            )
        return self.__problem_type

    def generate(self, random: Random) -> Problem:
        return self.evaluation.generate(random)


CREATE_SINGLE_EVALS_TABLE = """
create table if not exists single_evaluations(
    id integer primary key,
    evaluation_name text not null,
    chatbot_name text not null,
    problem_data text not null,
    ix integer not null,
    report_data text not null,
    unique (evaluation_name, chatbot_name, problem_data, ix)
)
"""


class BasicEvaluation(Evaluation[Problem]):
    def __init__(self, name: str | None = None):
        super().__init__(name=name)
        self.__problem_set = None

    def new_problem_set(self) -> ProblemSet[Problem]:
        if not hasattr(self, "generate"):
            raise TypeError("BasicEvaluation must define generate or new_problem_set")
        return BasicEvaluationProblemSet(self)

    @property
    def problem_set(self) -> ProblemSet[Problem]:
        if self.__problem_set is None:
            self.__problem_set = self.new_problem_set()
        assert self.__problem_set is not None
        return self.__problem_set

    @abstractmethod
    def run_single_evaluation(
        self, evaluation: SingleProblemEvaluation[Problem]
    ) -> None: ...

    def run(
        self,
        *,
        chatbot: Chatbot,
        random: Random | None = None,
        n_samples=1000,
        stop_on_first_failure=False,
        parallelism=1,
        reduce=False,
        pb=True,
        reraise_errors=False,
        storage=None,
    ) -> FullReport[Problem]:
        chatbot.freeze()
        if random is None:
            random = Random()

        assert random is not None

        if storage is None:
            storage = chatbot.storage

        pb_wrapper: Callable[[Any], Any]
        if pb:
            pb_wrapper = tqdm
        else:

            def pb_wrapper(x):
                return x

        with storage.cursor() as cursor:
            cursor.execute(CREATE_SINGLE_EVALS_TABLE)

        problem_set = self.problem_set
        report_adapter = TypeAdapter(SingleProblemReport[problem_set.problem_type])

        def evaluate_problem(
            problem: Problem, index: int = 0
        ) -> SingleProblemReport[Problem]:
            problem_data = problem_set.dump(problem)

            with storage.cursor() as cursor:
                cursor.execute(
                    "select report_data from single_evaluations "
                    "where evaluation_name = ? and chatbot_name = ? "
                    "and problem_data = ? and ix = ?",
                    (self.name, chatbot.name, problem_data, index),
                )

                for (result,) in cursor:
                    data = json.loads(result)
                    data["problem"] = problem
                    data["transcripts"] = [
                        storage.id_to_messages(t) for t in data["transcripts"]
                    ]
                    data["status"] = ReportedStatus[data["status"]]
                    data["errors"] = set(data["errors"])
                    return report_adapter.validate_python(data)

            problem_evaluation = SingleProblemEvaluation(
                chatbot=chatbot.clone(index=index),
                problem=problem,
                reraise_errors=reraise_errors,
            )
            try:
                self.run_single_evaluation(problem_evaluation)
            except NoValidAnswer:
                pass
            report = problem_evaluation.report()
            data_to_save = report_adapter.dump_python(report)
            data_to_save.pop("problem")
            data_to_save["transcripts"] = [
                storage.messages_to_transcript_id(t)
                for t in data_to_save["transcripts"]
            ]
            data_to_save["errors"] = sorted(data_to_save["errors"])
            data_to_save["status"] = data_to_save["status"].name
            with storage.cursor() as cursor:
                cursor.execute(
                    "insert into single_evaluations "
                    "(evaluation_name, chatbot_name, problem_data, ix, report_data) "
                    "values(?, ?, ?, ?, ?)",
                    (
                        self.name,
                        chatbot.name,
                        problem_data,
                        index,
                        json.dumps(data_to_save),
                    ),
                )

            return report

        if stop_on_first_failure:
            parallelism = 1

        if parallelism == 1:
            if stop_on_first_failure:
                initial_results = []
                for _ in pb_wrapper(range(n_samples)):
                    one_eval = evaluate_problem(self.problem_set.generate(random))
                    initial_results.append(one_eval)
                    if one_eval.status == ReportedStatus.INCORRECT:
                        break
            else:
                initial_results = [
                    evaluate_problem(self.problem_set.generate(random))
                    for _ in pb_wrapper(range(n_samples))
                ]
        else:
            with ThreadPoolExecutor(max_workers=parallelism) as executor:
                futures = [
                    executor.submit(evaluate_problem, self.problem_set.generate(random))
                    for _ in pb_wrapper(range(n_samples))
                ]
                initial_results = [future.result() for future in pb_wrapper(futures)]

        other_samples = []
        reduced_exemplar = None
        problem_to_report = {}

        if reduce:
            incorrect_answers = [
                r for r in initial_results if r.status == ReportedStatus.INCORRECT
            ]
            if incorrect_answers:
                incorrect_answers = list(incorrect_answers)
                target_confidence = max(
                    [r.confidence or 0.0 for r in incorrect_answers]
                )
                maximally_confident_incorrect_answers = [
                    r for r in incorrect_answers if r.confidence == target_confidence
                ]
                assert maximally_confident_incorrect_answers
                exemplar_problem = min(
                    maximally_confident_incorrect_answers,
                    key=lambda r: self.problem_set.reduction_key(r.problem),
                )

                def is_interesting(problem: Problem) -> bool:
                    report = evaluate_problem(problem)
                    other_samples.append(report)
                    problem_to_report[self.problem_set.hashable_key(problem)] = report
                    if report.status != ReportedStatus.INCORRECT:
                        return False
                    return (report.confidence or 0.0) >= target_confidence

                reduced_exemplar = problem_to_report[
                    self.problem_set.hashable_key(
                        self.problem_set.reduce(
                            exemplar_problem.problem, is_interesting
                        )
                    )
                ]

        return FullReport(
            model=chatbot.model,
            initial_random_sample=initial_results,
            other_samples=other_samples,
            reduced_exemplar=reduced_exemplar,
        )
