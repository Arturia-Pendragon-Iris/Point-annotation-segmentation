from scipy.ndimage import distance_transform_edt


def perform_distance_trans(np_array):
    assert len(np_array.shape) in [2, 3]
    return distance_transform_edt(1 - np_array)