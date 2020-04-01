import numpy as np
from h5py import File


def crop_center(data, target_shape):

    shp = np.array(data.shape)
    target_shp = np.array(target_shape)

    start = ((shp - target_shape) / 2).astype('int')

    return data[start[0]: start[0] + target_shp[0],
                start[1]: start[1] + target_shp[1],
                start[2]: start[2] + target_shp[2]
                ]

