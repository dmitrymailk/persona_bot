from typing import List
from itertools import chain


def flat_list(list_of_lists: List[List] | List) -> List:
    return list(chain.from_iterable(list_of_lists))
