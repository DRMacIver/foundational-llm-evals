from bisect import bisect_left
from collections.abc import Sequence
from functools import lru_cache

from foundationevals.data.files import DATA_DIR


class WordList(Sequence[str]):
    def __init__(self, words):
        self.__words = sorted(words)

    def __contains__(self, value):
        i = bisect_left(self.__words, value)
        return i < len(self.__words) and self.__words[i] == value

    def __iter__(self):
        yield from sorted(self.__words)

    def __len__(self):
        return len(self.__words)

    def __getitem__(self, i):
        return self.__words[i]

    def __repr__(self):
        return f"WordList({len(self.__words)} from {min(self.__words)} to {max(self.__words)})"


@lru_cache(maxsize=10)
def word_list(*names):
    assert names
    names = list(names)
    names[-1] += ".txt"
    file = DATA_DIR
    for n in names:
        file = file / n
    with file.open("r") as f:
        return WordList(f.read().splitlines())
