
import os
from paintera_multicut_workflow import pm_workflow


if __name__ == '__main__':

    # TODO: Set this!
    results_folder = '/path/to/results/'

    if not os.path.exists(results_folder):
        os.mkdir(results_folder)

    inputs_folder = '/g/schwab/hennies/FOR_CONSTANTIN'

    raw_filepath = os.path.join(
        inputs_folder,
        'raw/train2_x1390_y230_z561_pad.h5'
    )
    mem_filepath = os.path.join(
        inputs_folder,
        'mem/train2_x1390_y230_z561_pad.h5'
    )
    sv_filepath = os.path.join(
        inputs_folder,
        'sv/train2_x1390_y230_z561_pad.h5'
    )

    pm_workflow(
        results_folder=results_folder,
        raw_filepath=raw_filepath,
        mem_pred_filepath=mem_filepath,
        supervoxel_filepath=sv_filepath,
        mem_pred_channel=2,
        auto_crop_center=True,
        annotation_shape=(256, 256, 256),
        paintera_env_name='paintera_env_new',
        activation_command='source /home/hennies/miniconda3/bin/activate',
        export_binary=True,
        conncomp_on_paintera_export=True,
        verbose=True
    )
