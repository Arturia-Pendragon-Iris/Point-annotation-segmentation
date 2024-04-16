import numpy as np
from visualization.view_2D import plot_parallel
from hessian_matrix.hessian_2d import compute_hessian as h_2d
from hessian_matrix.hessian_3d import compute_hessian as h_3d

ct = np.load("/data/chest_CT/rescaled_ct/PS/ct_scan/ps000021.npz")["arr_0"][:100, :100, :100]
ct = np.clip((ct + 1000) / 1400, 0, 1)
l1, l2, l3 = h_3d(ct)

plot_parallel(
    a=l1[:, :, 50],
    b=l2[:, :, 50],
    c=l3[:, :, 50],
)