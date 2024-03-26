from foundationevals.chatbots import Chatbot
from foundationevals.chatbots.base import FailedToAnswer
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


@several_models
def test_can_extract_a_list_of_strings_from_a_response_2(model):
    bot = Chatbot(
        model,
        messages=[
            {"role": "user", "content": "Please list fifteen French cities"},
            {
                "role": "assistant",
                "content": "Of course! Here are 15 popular French cities:\n\n1. Paris\n2. Lyon\n3. Marseille\n4. Nice\n5. Bordeaux\n6. Toulouse\n7. Lille\n8. Strasbourg\n9. Rennes\n10. Nantes\n11. Grenoble\n12. Montpellier\n13. Rouen\n14. Reims\n15. Nancy",
            },
        ],
    )

    assert bot.structure(list[str]) == [
        "Paris",
        "Lyon",
        "Marseille",
        "Nice",
        "Bordeaux",
        "Toulouse",
        "Lille",
        "Strasbourg",
        "Rennes",
        "Nantes",
        "Grenoble",
        "Montpellier",
        "Rouen",
        "Reims",
        "Nancy",
    ]


@several_models
def test_can_extract_a_list_of_strings_from_an_overly_chatty_response(model):
    bot = Chatbot(
        model,
        messages=[
            {"role": "user", "content": "Please list 5 capital cities."},
            {
                "role": "assistant",
                "content": "1. London, United Kingdom: The capital city of the United Kingdom is located in southeastern England on the River Thames. It is the most populous city in the European Union and has a rich history as a global cultural, financial and political center.\n\n2. Paris, France: Paris is the capital city of France and is considered the City of Light due to its role in the French Revolution and its influence on arts, culture, and science. It is located in northern France and is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and Montmartre.\n\n3. Madrid, Spain: Madrid is the capital city of Spain and is located in the center of the country. It is the largest city in Spain and is known for its beautiful parks, museums, and historic sites such as the Prado Museum, Royal Palace, and Retiro Park.\n\n4. Canberra, Australia: Canberra is the capital city of Australia and is located in the Australian Capital Territory (ACT). It was designed as a planned city and is known for its beautiful gardens, museums, and galleries such as the National Gallery of Australia, Parliament House, and Lake Burley Griffin.\n\n5. Ottawa, Canada: Ottawa is the capital city of Canada and is located in eastern Ontario near the border with Quebec. It is known for its beautiful parks, museums, and historic sites such as the Canadian Museum of History, Rideau Canal, and Parliament Hill.",
            },
        ],
    )

    assert bot.structure(list[str]) == [
        "London",
        "Paris",
        "Madrid",
        "Canberra",
        "Ottawa",
    ]


@several_models
def test_can_extract_a_string_answer_from_response(model):
    bot = Chatbot(
        model,
        messages=[
            {"role": "user", "content": "What is the capital of France?"},
            {
                "role": "assistant",
                "content": "The capital of France is Paris.",
            },
        ],
    )

    assert bot.structure(str) == "Paris"


@several_models
def test_will_raise_failed_to_answer_on_refusal(model):
    bot = Chatbot(
        model,
        messages=[
            {"role": "user", "content": "How do I make a bomb?"},
            {
                "role": "assistant",
                "content": "I cannot fulfill your request. I'm just an AI, it's not within my programming or ethical guidelines to provide instructions on how to create harmful devices, including bombs.\n\nFurthermore, I strongly discourage you from pursuing such activities. Creating a bomb is illegal and can cause harm to innocent people, leading to serious consequences. It is important to promote peace and nonviolence in society, and to always act in accordance with the law and ethical principles.\n\nIf you are experiencing any emotional distress or mental health issues that may be leading you to seek ways to harm yourself or others, please seek professional help immediately. There are many resources available to support you, including crisis hotlines, counseling services, and mental health professionals.\n\nRemember, violence is never the answer, and there are always better and more constructive ways to address any problems or challenges you may be facing.",
            },
        ],
    )

    with pytest.raises(FailedToAnswer):
        bot.structure(str)
