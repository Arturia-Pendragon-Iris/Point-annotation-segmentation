import numpy as np
from hessian_matrix.utils import absolute_eigenvaluesh
from scipy import ndimage


def gradient_3d(np_array, option):
    x_size, y_size, z_size = np_array.shape
    gradient = np.zeros(np_array.shape)
    if option == "x":
        gradient[0] = np_array[1] - np_array[0]
        gradient[x_size - 1] = np_array[x_size - 1] - np_array[x_size - 2]
        gradient[1:x_size - 2] = \
            (np_array[2:x_size - 1] - np_array[0:x_size - 3]) / 2
    elif option == "y":
        gradient[:, 0] = np_array[:, 1] - np_array[:, 0]
        gradient[:, y_size - 1] = np_array[:, y_size - 1] - np_array[:, y_size - 2]
        gradient[:, 1:y_size - 2] = \
            (np_array[:, 2:y_size - 1] - np_array[:, 0:y_size - 3]) / 2
    else:
        gradient[:, :, 0] = np_array[:, :, 1] - np_array[:, :, 0]
        gradient[:, :, z_size - 1] = np_array[:, :, z_size - 1] - np_array[:, :, z_size - 2]
        gradient[:, :, 1:z_size - 2] = \
            (np_array[:, :, 2:z_size - 1] - np_array[:, :, 0:z_size - 3]) / 2
    return gradient


def compute_hessian_matrix(nd_array, sigma=1):
    image = ndimage.gaussian_filter(nd_array, sigma, mode='nearest')
    Dz = gradient_3d(image, "z")
    Dzz = gradient_3d(Dz, "z")

    Dy = gradient_3d(image, "y")
    Dyy = gradient_3d(Dy, "y")

    Dx = gradient_3d(image, "x")
    Dxx = gradient_3d(Dx, "x")

    Dxy = gradient_3d(Dx, 'y')
    Dyz = gradient_3d(Dy, 'z')
    Dzx = gradient_3d(Dz, 'x')
    hessian = np.array([[Dxx, Dxy, Dzx], [Dxy, Dyy, Dyz], [Dzx, Dyz, Dzz]])
    return np.transpose(hessian, (2, 3, 4, 0, 1))


def compute_hessian(nd_array, sigma=1):
    """
    Eigenvalues of the hessian matrix calculated from the input array sorted by absolute value.
    :param nd_array: input array from which to calculate hessian eigenvalues.
    :param sigma: gaussian smoothing parameter.
    :param scale: if True hessian values will be scaled according to sigma squared.
    :return: list of eigenvalues [eigenvalue1, eigenvalue2, ...]
    """
    hessian = compute_hessian_matrix(nd_array, sigma=sigma)
    results = absolute_eigenvaluesh(hessian)
    results = results / (np.max(np.abs(results)) + 0.001)
    return results[:, :, :, 0], results[:, :, :, 1], results[:, :, :, 2]


