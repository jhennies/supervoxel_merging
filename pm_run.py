import argparse

import os
from paintera_multicut_workflow import pm_workflow


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('result_folder', type=str)
    parser.add_argument('--raw', type=str, default=None)
    parser.add_argument('--mem', type=str, default=None)
    parser.add_argument('--sv', type=str, default=None)
    parser.add_argument('--paintera_env_name', type=str, default='paintera-env')
    parser.add_argument('--activation_command', type=str, default='conda activate')
    parser.add_argument('--annotation_shape', type=int, default=(256, 256, 256), nargs=3)
    parser.add_argument('--mem_channel', type=int, default=None)
    parser.add_argument('--verbose', type=bool, default=False)

    args = parser.parse_args()
    results_folder = args.result_folder
    raw_filepath = args.raw
    mem_filepath = args.mem
    sv_filepath = args.sv
    paintera_env_name = args.paintera_env_name
    activation_command = args.activation_command
    annotation_shape = args.annotation_shape
    mem_pred_channel = args.mem_channel
    verbose = args.verbose

    if verbose:
        print(args)

    if not os.path.exists(results_folder):
        os.mkdir(results_folder)

    pm_workflow(
        results_folder=results_folder,
        raw_filepath=raw_filepath,
        mem_pred_filepath=mem_filepath,
        supervoxel_filepath=sv_filepath,
        mem_pred_channel=mem_pred_channel,
        auto_crop_center=True,
        annotation_shape=annotation_shape,
        paintera_env_name=paintera_env_name,
        activation_command=activation_command,
        export_binary=True,
        conncomp_on_paintera_export=True,
        verbose=True
    )
