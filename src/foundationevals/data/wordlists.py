from functools import lru_cache

from foundationevals.data.files import DATA_DIR


@lru_cache(maxsize=10)
def word_list(*names):
    assert names
    names = list(names)
    names[-1] += ".txt"
    file = DATA_DIR
    for n in names:
        file = file / n
    with file.open("r") as f:
        return frozenset(f.read().splitlines())
