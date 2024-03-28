from foundationevals.data.wordlists import word_list
from abc import ABC, abstractmethod
from random import Random
from humanize import ordinal
from pydantic import BaseModel, field_validator, ValidationError
from foundationevals.evaluations.evaluation import (
    BadData,
    ProblemSet,
    SingleProblemEvaluation,
)
import json


class Rule(ABC, BaseModel):
    @abstractmethod
    def matches(self, word: str) -> bool: ...


class Length(Rule):
    min_length: int
    max_length: int

    def matches(self, word: str) -> bool:
        return self.min_length <= len(word) <= self.max_length


class Character(Rule):
    character: str
    position: int

    def matches(self, word: str) -> bool:
        try:
            return word[self.position].lower() == self.character.lower()
        except IndexError:
            return False


class WordConstraints(BaseModel):
    rules: list[Rule]

    def matches(self, word: str) -> bool:
        return all(rule.matches(word) for rule in self.rules)

    @field_validator("rules")
    @classmethod
    def rules_must_be_inhabited(cls, rules):
        words = word_list("sowpods")
        for i, rule in enumerate(rules):
            words = [word for word in words if rule.matches(word)]
            if not words:
                raise ValueError(
                    f"No words match all of {', '.join(map(repr, rules[:i+1]))}"
                )
        return rules


class WordConstraintsProblemSet(ProblemSet[WordConstraints]):
    def generate(self, random: Random) -> WordConstraints:
        working = list(word_list("sowpods"))
        rules = []
        while len(working) > 1:
            target1, target2 = random.sample(working, 2)
            if len(target1) != len(target2):
                if len(target1) < len(target2):
                    new_rule = Length(
                        min_length=len(target1), max_length=len(target2) - 1
                    )
                else:
                    new_rule = Length(
                        min_length=len(target2) + 1, max_length=len(target1)
                    )
            else:
                indices = [i for i in range(len(target1)) if target1[i] != target2[i]]
                assert indices
                i = random.choice(indices)
                new_rule = Character(character=target1[i], position=i)
            assert new_rule.matches(target1)
            rules.append(new_rule)
            working = [word for word in working if new_rule.matches(word)]
        random.shuffle(rules)
        n = random.randint(1, len(rules))
        return WordConstraints(rules=rules[:n])

    def load(self, data: bytes) -> WordConstraints:
        try:
            parsed = json.loads(data)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise BadData()

        if not isinstance(parsed, dict) or list(parsed) != ["rules"]:
            raise BadData(parsed)

        if not isinstance(parsed["rules"], list):
            raise BadData(parsed)

        rules = []
        try:
            for rule in parsed["rules"]:
                if not isinstance(rule, dict):
                    raise BadData(parsed)
                if "min_length" in rule:
                    rules.append(Length(**rule))
                else:
                    rules.append(Character(**rule))
            return WordConstraints(rules=rules)
        except ValidationError:
            raise BadData()

    def dump(self, problem: WordConstraints) -> bytes:
        return json.dumps({"rules": [r.model_dump() for r in problem.rules]}).encode(
            "utf-8"
        )


def words_matching_rules(rules):
    words = word_list("sowpods")
    for rule in rules:
        words = [word for word in words if rule.matches(word)]
    return words


def evaluate_ability_to_generate_words(
    evaluation: SingleProblemEvaluation[WordConstraints],
) -> None:
    sowpods = word_list("sowpods")
    rules = evaluation.problem.rules
    prompt_parts = [
        f"Please give me a list of words satisfying the following condition{'s' if len(rules) > 1 else ''}:\n"
    ]
    for rule in evaluation.problem.rules:
        if isinstance(rule, Length):
            if rule.min_length == rule.max_length:
                prompt_parts.append(
                    f"It should have exactly {rule.min_length} characters."
                )
            else:
                prompt_parts.append(
                    f"It should have between {rule.min_length} and {rule.max_length} characters."
                )
        elif isinstance(rule, Character):
            if rule.position == 0:
                prompt_parts.append(
                    f"It should start with the character {rule.character}."
                )
            elif rule.position == -1:
                prompt_parts.append(
                    f"It should end with the character {rule.character}."
                )
            elif rule.position < 0:
                pos = ordinal(-rule.position)
                prompt_parts.append(
                    f"The {pos} character from the end should be {rule.character}."
                )
            else:
                pos = ordinal(rule.position + 1)
                prompt_parts.append(f"The {pos} character should be {rule.character}.")

    prompt = "\n".join(prompt_parts)

    evaluation.chatbot.chat(prompt)
    results = evaluation.parse(list[str])

    if not results:
        evaluation.add_error("No words were given.")
        return

    for word in results:
        if word.upper() not in sowpods:
            evaluation.add_note(f"{word} is not in the SOWPODS word list.")
        for rule in evaluation.problem.rules:
            if not rule.matches(word):
                evaluation.add_note(f"Word {word} does not match the rule {rule}")
                if isinstance(rule, Character):
                    evaluation.add_error("Returned a word with an incorrect character.")
                else:
                    assert isinstance(rule, Length)
                    evaluation.add_error("Returned a word of the wrong length")
