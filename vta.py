import numpy as np
from scipy.linalg import toeplitz
from scipy.linalg import inv


def vta_huang(data: np.ndarray, *, n_tube: int = 21, window_length: int = 147) -> np.ndarray:
    x = np.hsplit(data, np.arange(window_length, len(data), window_length))
    area = np.ones((n_tube + 1, len(x)))

    for i, frame in enumerate(x):
        mat = _sum_of_mat_acr(frame, n_tube)
        y = _sum_of_y_part(frame, n_tube)
        k = inv(mat, overwrite_a=True).dot(y)
        for m in range(n_tube):
            area[m+1, i] = area[m, i] * (1 - k[m]) / (1 + k[m])
    return np.absolute(_normalize(area[1:, :]))


def vta_paper(data: np.ndarray, n_tube: int = 21, window_length: int = 147) -> np.ndarray:
    x = np.hsplit(data, np.arange(window_length, len(data), window_length))
    area = np.ones((n_tube + 1, len(x)))

    for i, frame in enumerate(x):
        y, mat = _sum_of_both_part_r(frame, n_tube)
        k = inv(mat, overwrite_a=True).dot(y)
        for m in range(n_tube):
            area[m+1, i] = area[m, i] * (1 - k[m]) / (1 + k[m])
    return np.absolute(_normalize(area[1:, :]))


def _normalize(v):
    try:
        return v / np.linalg.norm(v, axis=0)
    except ZeroDivisionError:
        return v


def _sum_of_mat_acr(data, tube_num):
    result = np.zeros([tube_num, tube_num])
    for i in range(1, tube_num + 1):
        for k in range(1, tube_num + 1):
            m = max(i, k)
            result[i-1][k-1] = (data[m-i:-i] * data[m-k:-k]).sum()
    return result


def _sum_of_y_part(data, tube_num):
    result = np.zeros(tube_num)
    for i in range(1, tube_num + 1):
        result[i-1] = (data[i:] * data[:-i]).sum()
    return result


def _sum_of_both_part_r(data, n_tube):
    result = np.zeros(n_tube + 1)
    for i in range(n_tube + 1):
        result[i] = (data[:len(data)-i] * data[i:]).sum()
    return result[1:], toeplitz(result[:-1])
