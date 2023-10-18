import numpy as np
from numba import njit


def Stats(data: np.array) -> (np.float64, np.float64):
    data_copy = data
    mean = np.sum(data_copy)/data_copy.size
    var = np.var(data_copy, ddof=1)

    return mean, var


def FullStats(data: np.array
              ) -> (np.array, np.array):

    mean_vector = np.zeros(data.size, dtype=np.float64)
    std_vector = np.zeros(data.size, dtype=np.float64)
    for i in range(data.size):
        mean_vector[i] = np.sum(data[:i+1])/data[:i+1].size
        std_vector[i] = np.std(data[:i+1], ddof=1)

    return mean_vector, std_vector
