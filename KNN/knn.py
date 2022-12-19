from math import *
from typing import List, Callable

import numpy as np
from numpy.typing import NDArray


def most_common(lst: List):
    return max(set(lst), key=lst.count)


def euclidean_distance(n: NDArray[float], m: NDArray[float]) -> float:
    return sqrt(np.sum((n - m) ** 2))


def manhattan_distance(n: NDArray[float], m: NDArray[float]) -> float:
    return np.sum(np.abs(n - m))


def k_distance(k: int) -> Callable[[NDArray[float], NDArray[float]], float]:
    def function(n: NDArray[float], m: NDArray[float]) -> float:
        return pow(np.sum(np.abs(n - m) ** k), 1 / k)

    return function


class KNNClassifier:
    def __init__(self, k: int, distance=euclidean_distance, ):
        self.k = k
        self.distance_function = distance

    def train(self, X: NDArray[NDArray[float]], Y: NDArray) -> None:
        self.X = X
        self.Y = Y

    def __find_k_nearest_neighbor(self, p: NDArray[float]) -> NDArray[int]:
        distances = [
            (neighbor_index, self.distance_function(self.X[neighbor_index], p))
            for neighbor_index in range(len(self.X))
        ]
        distances.sort(key=lambda x: x[1])
        return [x[0] for x in distances[:self.k]]

    def predicate_single(self, p: NDArray[float]):
        nearest_points_indices = self.__find_k_nearest_neighbor(p)
        classes = [self.Y[index] for index in nearest_points_indices]
        return most_common(classes)

    def predicate(self, p: NDArray[NDArray[float]]):
        return [self.predicate_single(instance) for instance in p]

# knn = KNNClassifier(3)
# knn.train(np.array(
#     [
#         [0, 1],
#         [0, 2],
#         [0, 3],
#         [1, 0],
#         [2, 0],
#         [3, 0],
#     ]
# ), np.array([
#     1, 1, 1, 0, 0, 0
# ])
# )
#
# print(knn.predicate_single(np.array([0.5, 0.4])))
