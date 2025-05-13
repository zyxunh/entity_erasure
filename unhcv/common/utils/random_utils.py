# Author: zhuyixing zhuyixing@bytedance.com
# Date: 2023-02-22 03:12:47
# LastEditors: zhuyixing zhuyixing@bytedance.com
# Description:
import numpy as np


__all__ = ["random_choice", "uniform", "RandomChoiceFlag"]


def random_choice(x, p=None):
    return x[np.random.choice(np.arange(len(x)), p=p)]


def uniform(probs):
    if isinstance(probs[0], (list, tuple)):
        if len(probs[0]) == 3:
            prob = random_choice(probs, p=[var[2] for var in probs])[:2]
        else:
            prob = random_choice(probs)
    else:
        prob = probs

    return np.random.uniform(*prob)


class RandomChoiceFlag:
    def __init__(self, probabilities):
        if not isinstance(probabilities, np.ndarray):
            probabilities = np.array(probabilities)
        probabilities = probabilities / probabilities.sum()
        self.probabilities_cumsum = np.cumsum(probabilities)
        self.random_value = np.random.random()
        self.i = -1

    def __call__(self):
        self.i += 1
        return self.random_value < self.probabilities_cumsum[self.i]