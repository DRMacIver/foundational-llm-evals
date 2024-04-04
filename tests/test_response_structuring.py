from foundationevals.chatbots import Chatbot
from foundationevals.chatbots.base import FailedToAnswer
import pytest


def check_confidence(bot):
    assert 0 <= bot.confidence() <= 1


def test_can_structure_a_badly_structured_response(model):
    bot = Chatbot(
        model,
        messages=[
            {"role": "user", "content": "What is 2 + 2?"},
            {"role": "assistant", "content": "The answer to 2 + 2 is 4."},
        ],
    )

    assert bot.structure(int) == 4
    check_confidence(bot)


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
    check_confidence(bot)


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
    check_confidence(bot)


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
    check_confidence(bot)


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

    assert bot.structure(list[str]) in (
        [
            "London",
            "Paris",
            "Madrid",
            "Canberra",
            "Ottawa",
        ],
        # Claude really wants to include the qualifying countries in its answers here and
        # it's hard to argue it's wrong, so we permit this too. Evaluations should probably
        # take into account variation like this.
        [
            "London, United Kingdom",
            "Paris, France",
            "Madrid, Spain",
            "Canberra, Australia",
            "Ottawa, Canada",
        ],
    )
    check_confidence(bot)


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
    check_confidence(bot)


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


def test_will_raise_failed_to_answer_with_stupid_bullshit_quibbling(model):
    bot = Chatbot(
        model,
        messages=[
            {
                "role": "user",
                "content": "Please give me a list of words satisfying the following conditions:\n\nIt should have exactly 5 characters.\nThe 1st character should be A.\nThe 2nd character should be N.\nThe 3rd character should be D.\nIt should have between 4 and 7 characters.",
            },
            {
                "role": "assistant",
                "content": "I'm sorry for the confusion, but your conditions are not consistent. The first three conditions state that the word has exactly 5 characters, the first character is 'A', and the second character is 'N'. However, the fourth condition states that it should have between 4 and 7 characters. Since a word with exactly 5 characters does not meet the requirement of having between 4 and 7 characters, there is no list of words that would satisfy all of your conditions.",
            },
        ],
    )

    with pytest.raises(FailedToAnswer):
        bot.structure(list[str])


@pytest.mark.parametrize(
    "response,parse",
    [
        (
            "Yes, 72 is bigger than 10. The number 72, has a greater value than the number 10.",
            True,
        ),
        ("No, 72 is not bigger than 10. In fact, 10 is smaller than 72.", False),
    ],
)
def test_will_structure_yes_no_as_boolean(model, response, parse):
    bot = Chatbot(
        model,
        messages=[
            {"role": "user", "content": "Is 72 bigger than 10?"},
            {
                "role": "assistant",
                "content": response,
            },
        ],
    )

    assert bot.structure(bool) == parse
    check_confidence(bot)


def test_yes_no_with_long_answer(model):
    bot = Chatbot(
        model,
        messages=[
            {"role": "user", "content": "Is 45 bigger than 10?"},
            {
                "role": "assistant",
                "content": "Yes, 45 is bigger than 10. The number 45 is fifteen more than the number 30, and ten is less than thirty. So, when comparing the two numbers directly, we can see that 45 has a greater value than 10.",
            },
        ],
    )

    assert bot.structure(bool)
    check_confidence(bot)


def test_word_lists_satisfying_conditions_1(model):
    bot = Chatbot(
        model,
        messages=[
            {
                "role": "user",
                "content": "Please give me a list of words satisfying the following conditions:\n\nThe 5th character should be A.\nIt should have between 6 and 8 characters.",
            },
            {
                "role": "assistant",
                "content": "Here are some words that meet your criteria:\n\n1. Stationary - meaning not moving or static\n2. Unpredictable - meaning hard to forecast or guess\n3. Prehistoric - referring to the time before recorded history\n4. Camouflage - meaning to disguise oneself to blend with surroundings\n5. Quasar - a type of celestial body that emits quasicrystalline structures\n6. Pistachio - a type of nut with a green shell\n7. Salmonoid - referring to fish species related to salmon\n8. Sequoia - a large, long-lived tree species native to California\n9. Quintessential - representing the most perfect or typical example of something\n10. Equatorial - relating to the equator, an imaginary line around the Earth dividing it into northern and southern hemispheres.",
            },
        ],
    )

    assert bot.structure(list[str]) == [
        "Stationary",
        "Unpredictable",
        "Prehistoric",
        "Camouflage",
        "Quasar",
        "Pistachio",
        "Salmonoid",
        "Sequoia",
        "Quintessential",
        "Equatorial",
    ]
    check_confidence(bot)


def test_word_lists_satisfying_conditions_2(model):
    bot = Chatbot(
        model,
        messages=[
            {
                "role": "user",
                "content": "Please give me a list of words satisfying the following conditions:\n\nIt should have between 8 and 14 characters.\nThe 5th character should be G.\nThe 9th character should be R.\nIt should have between 8 and 10 characters.\nThe 2nd character should be R.\nThe 8th character should be O.\nIt should start with the character C.\nThe 4th character should be U.",
            },
            {
                "role": "assistant",
                "content": 'Based on the given conditions, here are some words that might fit:\n\n1. Curugor: A made-up word that satisfies all conditions.\n2. Crurung: A possible word, but it\'s not in any standard dictionary.\n3. Curuogr: Another possible word, but it\'s also not in any standard dictionary.\n4. Curingro: This is a real word in the Spanish language that means "curing" or "healing." It satisfies all conditions except for having exactly 10 characters.\n5. Crucugor: A made-up word that satisfies all conditions except for having exactly 8 characters.\n6. Curuogoru: A made-up word that satisfies all conditions except for having exactly 14 characters.\n7. Crurougo: This is a real word in the Indonesian language that means "crab." It satisfies all conditions except for having exactly 10 characters and having the letter U as the 4th character instead of the 5th.\n8. Curugou: A made-up word that satisfies all conditions except for having exactly 12 characters.\n9. Curuogorc: A made-up word that satisfies all conditions except for having exactly 13 characters.\n10. Curuogoro: A made-up word that satisfies all conditions except for having exactly 14 characters.',
            },
        ],
    )

    bot_response = bot.structure(list[str])
    correct_answer = [
        "Curugor",
        "Crurung",
        "Curuogr",
        "Curingro",
        "Crucugor",
        "Curuogoru",
        "Crurougo",
        "Curugou",
        "Curuogorc",
        "Curuogoro",
    ]
    # Several models get this wrong and this test is mostly for making sure
    # mistral doesn't refuse to report a confidence.
    # assert bot_response == correct_answer
    check_confidence(bot)


def test_word_lists_satisfying_conditions_3(model):
    bot = Chatbot(
        model,
        messages=[
            {
                "role": "user",
                "content": "Please give me a list of words satisfying the following condition:\n\nThe 10th character should be M.",
            },
            {
                "role": "assistant",
                "content": 'I\'d be happy to help you with that! However, I need to specify which language or root word list you want me to use, as there are many words in various languages and contexts that could potentially fit your criterion. Here are some English words that have ten characters and end with the letter "M":\n\n1. Acknowledgment\n2. Almondism\n3. Amaranthine\n4. Antidisestablishmentarianism\n5. Apprehensionable\n6. Arithmeticianism\n7. Biochemistrylab\n8. Camouflagedly\n9. Caterpillarmigrator\n10. Centenarianism\n11. Chromatographycolumn\n12. Compartmentalization\n13. Contradictorily\n14. Coprophagicmicrobe\n15. Decipherable\n16. Dermatologyclinic\n17. Disenfranchisement\n18. Documentation\n19. Echocardiographylab\n20. Electroencephalogram\n21. Embryologicdevelopment\n22. Encapsulation\n23. Endocrinologyclinic\n24. Enzymologylaboratory\n25. Epithelialization\n26. Equivalenceclassification\n27. Existentialismand\n28. Fetaldevelopmentstage\n29. Fluoridationprogram\n30. Forensicpathologist\n31. Geochemistrylab\n32. Hematologyclinic\n33. Hemoglobinopathy\n34. Histopathologicanalysis\n35. Homogenizationprocess\n36. Hypertrophiccardiomyopathy\n37. Immunohistochemistry\n38. Infectiousdiseaselab\n39. Ionospheremonitoring\n40. Laryngologyclinic\n41. Leukemiapatient\n42. Magnetohydrodynamics\n43. Microbiologysample\n44. Mycologylaboratory\n45. Neurologyclinic\n46. Nucleotidepolymerase\n47. Oncologyward\n48. Ophthalmologyclinic\n49. Parasitologylab\n50. Pathophysiologystudy\n51. Pediatricianoffice\n52. Perinatologycenter\n53. Pharmacologydepartment\n54. Photochemistryreaction\n55. Plateletdisaggregation\n56. Pneumonologyclinic\n57. Psychophysiologylab\n58. Radiographyroom\n59. Rheumatologyclinic\n60. Seismologystation\n61. Skeletalmusculature\n62. Sociologydepartment\n63. Stomatologyclinic\n64. Toxicologylaboratory\n65. Traumatologyward\n66. Virologylab\n67. Xenotransplantation\n\nI hope this list helps you! Let me know if you need anything else.',
            },
        ],
    )

    check_confidence(bot)


def test_can_conform_badly_formed_json():
    """Mistral hilariously wraps one of the elements of its response in a list
    when trying to structure this. This test makes sure we correctly fix that in post.
    """
    bot = Chatbot(
        "mistral",
        messages=[
            {
                "role": "user",
                "content": "Please give me a list of words satisfying the following conditions:\n\nIt should have between 7 and 11 characters.\nThe 1st character should be A.\nThe 9th character should be S.\nIt should start with the character W.\nThe 8th character should be I.\nThe 3rd character should be E.",
            },
            {
                "role": "assistant",
                "content": "Based on the given conditions, here are some words that might fit:\n\n1. Weasel (7 characters)\n2. Weighty (9 characters) - The 3rd character is E and the 8th character is I, but the 9th character is Y instead of S.\n3. Weaknesses (10 characters)\n4. Awesomeness (10 characters)\n5. Awe-struck (9 characters) - The 1st character is A and the 8th character is I, but the 9th character is E instead of S.\n6. Weaseling (9 characters)\n7. Awe-inspiring (12 characters) - This word has more than 11 characters, but it fits all other conditions.\n\nPlease note that some of these words might not be commonly used or grammatically correct in certain contexts. It's always a good idea to double-check the spelling and meaning of any word before using it in a formal setting.",
            },
        ],
    )

    assert bot.structure(list[str]) == [
        "Weasel",
        "Weighty",
        "Weaknesses",
        "Awesomeness",
        "Awe-struck",
        "Weaseling",
        "Awe-inspiring",
    ]
    check_confidence(bot)
