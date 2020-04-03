import argparse

import os
from paintera_multicut_workflow import pm_workflow


if __name__ == '__main__':

    # TODO these should not be default args
    default_inputs_folder = '/home/pape/Work/data/julian/paintera_mc_wf_example_package'
    default_command = 'source /g/kreshuk/pape/Work/software/conda/miniconda3/bin/activate'

    parser = argparse.ArgumentParser()
    parser.add_argument('result_folder', type=str)
    parser.add_argument('--inputs_folder', type=str, default=default_inputs_folder)
    parser.add_argument('--paintera_env_name', type=str, default='paintera3')
    parser.add_argument('--activation_command', type=str, default=default_command)

    args = parser.parse_args()
    results_folder = args.result_folder
    inputs_folder = args.inputs_folder
    paintera_env_name = args.paintera_env_name
    activation_command = args.activation_command

    if not os.path.exists(results_folder):
        os.mkdir(results_folder)

    # TODO don't hard-code this, but also take as input args
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
        mem_pred_channel=None,
        auto_crop_center=True,
        annotation_shape=(256, 256, 256),
        paintera_env_name=paintera_env_name,
        activation_command=activation_command,
        export_binary=True,
        conncomp_on_paintera_export=True,
        verbose=True
    )
