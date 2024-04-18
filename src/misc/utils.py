import numpy as np
from collections import defaultdict


class EdgeKeyDict(dict):
    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = tuple(key)
        return super().__getitem__(key)


class DefaultList:
    def __init__(self):
        self.data = defaultdict(int)  # defaults to int, which is 0

    def __getitem__(self, index):
        return self.data[index]

    def __setitem__(self, index, value):
        self.data[index] = value

    def __repr__(self):
        # Optional: Create a list representation only for non-zero values for easier debugging
        if self.data:
            max_index = max(self.data.keys())
            return repr([self.data[i] for i in range(max_index + 1)])
        return repr([])


def mat2str(mat):
    return (
        str(mat)
        .replace("'", '"')
        .replace("(", "<")
        .replace(")", ">")
        .replace("[", "{")
        .replace("]", "}")
    )


def dictsum(dic, t):
    return sum([dic[key][t] for key in dic])


def moving_average(a, n=3):
    """
    Computes a moving average used for reward trace smoothing.
    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n
