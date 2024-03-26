from foundationevals.chatbots import Chatbot
import pytest


several_models = pytest.mark.parametrize("model", ["gemma", "llama2", "mistral"])


@several_models
def test_can_structure_a_badly_structured_response(model):
    bot = Chatbot(
        model,
        messages=[
            {"role": "user", "content": "What is 2 + 2?"},
            {"role": "assistant", "content": "The answer to 2 + 2 is 4."},
        ],
    )

    assert bot.structure(int) == 4


@several_models
def test_can_extract_a_list_of_ints_from_a_response(model):
    bot = Chatbot(
        model,
        messages=[
            {"role": "user", "content": "What are the first three prime numbers?"},
            {
                "role": "assistant",
                "content": "The first three prime numbers are 2, 3, and 5.",
            },
        ],
    )

    assert bot.structure(list[int]) == [2, 3, 5]


@several_models
def test_can_extract_a_list_of_strings_from_a_response(model):
    bot = Chatbot(
        model,
        messages=[
            {"role": "user", "content": "Please list three words starting with Q"},
            {
                "role": "assistant",
                "content": "Sure! Here are three words starting with the letter Q:\n\n1. Queen\n2. Question\n3. Quiet",
            },
        ],
    )

    assert bot.structure(list[str]) == ["Queen", "Question", "Quiet"]
