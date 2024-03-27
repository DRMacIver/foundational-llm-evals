from foundationevals.data.wordlists import word_list


def test_sowpods():
    assert "AAHED" in word_list("sowpods")
