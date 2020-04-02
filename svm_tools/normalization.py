
import numpy as np
import os
from h5py import File
from tifffile import imread, imsave
from multiprocessing import Pool
from glob import glob


def normalize_slices_with_quantiles(volume, quantile=0.05):

    dtype = volume.dtype
    assert dtype == 'uint8', 'Only unsigned 8bit is implemented'
    volume = volume.astype('float64')

    # Get quantiles of full volume
    # Could potentially also be a reference slice, multiple reference slices, ...
    q_lower_ref = np.quantile(volume, quantile)
    q_upper_ref = np.quantile(volume, 1 - quantile)

    # Process slices
    # This can be parallelized
    for slid, sl in enumerate(volume):

        # Get quantiles of the image slice
        q_lower = np.quantile(sl, quantile)
        q_upper = np.quantile(sl, 1 - quantile)

        # Convert the intensities to the target domain
        sl -= q_lower
        sl /= q_upper - q_lower
        sl *= q_upper_ref - q_lower_ref
        sl += q_lower_ref

        volume[slid] = sl

    # Clip everything that went out of range
    # FIXME this assumes dtype==uint8
    volume[volume < 0] = 0
    volume[volume > 255] = 255

    # Convert back to the original dtype
    return volume.astype(dtype)


def normalize_tif_with_quantiles(filepath, target_folder, q_lower_ref, q_upper_ref, quantile):

    sl = imread(filepath)
    target_filepath = os.path.join(
        target_folder,
        os.path.split(filepath)[1]
    )

    if not os.path.isfile(target_filepath):

        print('processing: {}'.format(target_filepath))

        assert sl.dtype == 'uint8'
        sl = sl.astype('float64')

        # Get quantiles of the image slice
        q_lower = np.quantile(sl, quantile)
        q_upper = np.quantile(sl, 1 - quantile)

        # Convert the intensities to the target domain
        sl -= q_lower
        sl /= q_upper - q_lower
        sl *= q_upper_ref - q_lower_ref
        sl += q_lower_ref

        # Clip everything that went out of range
        sl[sl < 0] = 0
        sl[sl > 255] = 255

        imsave(target_filepath, data=sl.astype('uint8'))

    else:

        print('exists: {}'.format(target_filepath))

def normalize_h5_with_quantiles(filepath, target_folder, q_lower_ref, q_upper_ref, quantile):

    with File(filepath, mode='r') as f:
        sl = f['data'][:]

    target_filepath = os.path.join(
        target_folder,
        os.path.split(filepath)[1]
    )

    if not os.path.isfile(target_filepath):

        print('processing: {}'.format(target_filepath))

        assert sl.dtype == 'uint8'
        sl = sl.astype('float64')

        # Get quantiles of the image slice
        q_lower = np.quantile(sl, quantile)
        q_upper = np.quantile(sl, 1 - quantile)

        # Convert the intensities to the target domain
        sl -= q_lower
        sl /= q_upper - q_lower
        sl *= q_upper_ref - q_lower_ref
        sl += q_lower_ref

        # Clip everything that went out of range
        sl[sl < 0] = 0
        sl[sl > 255] = 255

        with File(target_filepath, mode='w') as f:
            f.create_dataset('data', data=sl.astype('uint8'), compression='gzip')

    else:

        print('exists: {}'.format(target_filepath))


class QuantileNormalizer:

    def __init__(
            self,
            ref_data,
            quantile
    ):

        self.quantile = quantile
        self.q_lower = np.quantile(ref_data, quantile)
        self.q_upper = np.quantile(ref_data, 1 - quantile)

    def run_on_tif_stack(self, folder, target_folder, n_workers=1):

        im_list = np.sort(glob(os.path.join(folder, '*.tif')))

        if n_workers == 1:
            [normalize_tif_with_quantiles(fp, target_folder, self.q_lower, self.q_upper, self.quantile) for fp in im_list]

        else:
            print('{} workers'.format(n_workers))
            with Pool(processes=n_workers) as p:
                tasks = [
                    p.apply_async(
                        normalize_tif_with_quantiles, (
                            fp, target_folder, self.q_lower, self.q_upper, self.quantile
                        )
                    )
                    for fp in im_list
                ]
            [task.get() for task in tasks]

    def run_on_h5(self, folder, target_folder):

        im_list = np.sort(glob(os.path.join(folder, '*.h5')))

        [normalize_h5_with_quantiles(fp, target_folder, self.q_lower, self.q_upper, self.quantile) for fp in im_list]

