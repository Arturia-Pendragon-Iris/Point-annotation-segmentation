import numpy as np
from scipy import ndimage


def gradient_2d(np_array, option):
    x_size = np_array.shape[0]
    y_size = np_array.shape[1]
    gradient = np.zeros(np_array.shape)
    if option == "x":
        gradient[0, :] = np_array[1, :] - np_array[0, :]
        gradient[x_size - 1, :] = np_array[x_size - 1, :] - np_array[x_size - 2, :]
        gradient[1:x_size - 2, :] = \
            (np_array[2:x_size - 1, :] - np_array[0:x_size - 3, :]) / 2
    else:
        gradient[:, 0] = np_array[:, 1] - np_array[:, 0]
        gradient[:, y_size - 1] = np_array[:, y_size - 1] - np_array[:, y_size - 2]
        gradient[:, 1:y_size - 2] = \
            (np_array[:, 2:y_size - 1] - np_array[:, 0:y_size - 3]) / 2
    return gradient


def hessian2d(image, sigma):
    # print(sigma)
    image = ndimage.gaussian_filter(image, sigma, mode='nearest')
    Dy = gradient_2d(image, "y")
    Dyy = gradient_2d(Dy, "y")

    Dx = gradient_2d(image, "x")
    Dxx = gradient_2d(Dx, "x")
    Dxy = gradient_2d(Dx, 'y')
    return Dxx, Dyy, Dxy


def eigval_hessian2d(Dxx, Dyy, Dxy):
    tmp = np.sqrt((Dxx - Dyy) ** 2 + 4 * (Dxy ** 2))
    mu1 = 0.5 * (Dxx + Dyy + tmp)
    mu2 = 0.5 * (Dxx + Dyy - tmp)

    indices = (np.absolute(mu1) > np.absolute(mu2))
    Lambda1 = mu1
    Lambda1[indices] = mu2[indices]

    Lambda2 = mu2
    Lambda2[indices] = mu1[indices]
    return Lambda1, Lambda2


def compute_hessian(img, sigma=1):
    hxx, hyy, hxy = hessian2d(img, sigma)

    c = sigma ** 2
    hxx = -c * hxx
    hyy = -c * hyy
    hxy = -c * hxy

    B1 = -(hxx + hyy)
    B2 = hxx * hyy - hxy ** 2
    T = np.ones(B1.shape)
    T[(B1 < 0)] = 0
    T[(B1 == 0) & (B2 == 0)] = 0
    T = T.flatten()
    indeces = np.where(T == 1)[0]
    hxx = hxx.flatten()
    hyy = hyy.flatten()
    hxy = hxy.flatten()
    hxx = hxx[indeces]
    hyy = hyy[indeces]
    hxy = hxy[indeces]

    lambda1i, lambda2i = eigval_hessian2d(hxx, hyy, hxy)
    lambda1 = np.zeros(img.shape[0] * img.shape[1], )
    lambda2 = np.zeros(img.shape[0] * img.shape[1], )

    lambda1[indeces] = lambda1i
    lambda2[indeces] = lambda2i

    # removing noise
    lambda1[(np.isinf(lambda1))] = 0
    lambda2[(np.isinf(lambda2))] = 0

    lambda1[(np.absolute(lambda1) < 1e-4)] = 0
    lambda1 = lambda1.reshape(img.shape)

    lambda2[(np.absolute(lambda2) < 1e-4)] = 0
    lambda2 = lambda2.reshape(img.shape)
    return (lambda1 / (np.max(np.abs(lambda1)) + 0.001),
            lambda2 / (np.max(np.abs(lambda1)) + 0.001))