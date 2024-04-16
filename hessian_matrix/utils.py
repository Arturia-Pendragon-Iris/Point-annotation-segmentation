import numpy as np
from visualization.view_2D import plot_parallel


def divide_nonzero(array1, array2):
    """
    Divides two arrays. Returns zero when dividing by zero.
    """
    denominator = np.copy(array2)
    denominator[denominator == 0] = 1e-10
    return np.divide(array1, denominator)


def create_image_like(data, image):
    return image.__class__(data, affine=image.affine, header=image.header)


def absolute_eigenvaluesh(nd_array):
    """
    Computes the eigenvalues sorted by absolute value from the symmetrical matrix.
    :param nd_array: array from which the eigenvalues will be calculated.
    :return: A list with the eigenvalues sorted in absolute ascending order (e.g. [eigenvalue1, eigenvalue2, ...])
    """
    eigenvalues = np.linalg.eigvalsh(nd_array)
    # print(eigenvalues.shape)
    sorted_eigenvalues = sortbyabs(eigenvalues, axis=-1)

    return sorted_eigenvalues


def sortbyabs(a, axis=-1):
    # print(a)
    index = np.argsort(np.abs(a), axis=axis)
    sorted_a = np.take_along_axis(a, index, axis=axis)
    # print(sorted_a)
    return sorted_a

