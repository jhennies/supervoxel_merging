
import numpy as np


# Determines whether a position was selected that is in the exclusion zone
def _position_in_exclusion_zone(pos, shp, exclusion_zone):
    if pos is None:
        return True
    if exclusion_zone is not None:
        count = 0
        for didx, d in enumerate(exclusion_zone):
            if d.start is not None:
                if pos[didx] > d.start - shp[didx]:
                    count += 1
            else:
                count += 1
            if d.stop is not None:
                if pos[didx] < d.stop:
                    count += 1
            else:
                count += 1
        if count == 6:
            return True

    return False


def build_equally_spaced_volume_list(
        shape,
        target_shape=(64, 64, 64),
        subvolume_start_index=(0, 0, 0),
        overlap=(0, 0, 0),
        exclusion_zone=None,
        overshoot=False
):

    # Components
    full_shape = shape
    if subvolume_start_index is not None:
        shape = np.array(full_shape) - np.array(subvolume_start_index)
    else:
        shape = np.array(full_shape)

    index_array = []

    # This generates the list of all positions assuming no transformations
    if overshoot:
        mg = np.mgrid[
             0: shape[0] - 1: target_shape[0] - overlap[0],
             0: shape[1] - 1: target_shape[1] - overlap[1],
             0: shape[2] - 1: target_shape[2] - overlap[2]
             ].squeeze()
    else:
        mg = np.mgrid[
             0: shape[0] - target_shape[0] + 1: target_shape[0] - overlap[0],
             0: shape[1] - target_shape[1] + 1: target_shape[1] - overlap[1],
             0: shape[2] - target_shape[2] + 1: target_shape[2] - overlap[2]
             ].squeeze()
    mg = mg.reshape(3, np.prod(np.array(mg.shape)[1:]))
    if subvolume_start_index is not None:
        positions = mg.swapaxes(0, 1) + subvolume_start_index
    else:
        positions = mg.swapaxes(0, 1)

    for position in positions:

        if not _position_in_exclusion_zone(position, target_shape, exclusion_zone):

            index_array.append(
                [
                    np.s_[:],
                    np.s_[position[0]: position[0] + target_shape[0],
                    position[1]: position[1] + target_shape[1],
                    position[2]: position[2] + target_shape[2]],
                    False
                ]
            )

    return len(index_array), index_array


def write_test_h5_generator_result(dataset, result, x, y, z, overlap, ndim=3):

    size = result.shape[1: ndim+1]
    if ndim == 2:
        size = (1,) + size
    dataset_shape = dataset.shape

    p = (z, y, x)

    s_ds = []
    s_r = []

    for idx in range(0, 3):
        zds = []
        zr = []
        if p[idx] == 0:
            zds.append(0)
            zr.append(0)
        else:
            zds.append(int(p[idx] + overlap[idx] / 2))
            zr.append(int(overlap[idx] / 2))

        if p[idx] + size[idx] == dataset_shape[idx]:
            zds.append(None)
            zr.append(None)
        else:
            zds.append(int(p[idx] + size[idx] - overlap[idx] / 2))
            zr.append(-int(overlap[idx] / 2))
            if zr[-1] == 0:
                zr[-1] = None

        s_ds.append(zds)
        s_r.append(zr)

    result = result.squeeze()
    if ndim == 2:
        result = result[None, :]

    if result.ndim == 3:
        # Just one channel

        dataset[s_ds[0][0]:s_ds[0][1], s_ds[1][0]:s_ds[1][1], s_ds[2][0]:s_ds[2][1]] \
            = result[s_r[0][0]:s_r[0][1], s_r[1][0]:s_r[1][1], s_r[2][0]:s_r[2][1]]

    elif result.ndim == 4:
        # Multiple channels

        dataset[s_ds[0][0]:s_ds[0][1], s_ds[1][0]:s_ds[1][1], s_ds[2][0]:s_ds[2][1], :] \
            = result[s_r[0][0]:s_r[0][1], s_r[1][0]:s_r[1][1], s_r[2][0]:s_r[2][1], :]

