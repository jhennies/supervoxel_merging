import argparse

import os
from paintera_multicut_workflow import pm_workflow
import getpass
import json


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--result_folder', type=str, default=None)
    parser.add_argument('--input_folder', type=str, default=None)
    parser.add_argument('--paintera_env_name', type=str, default=None)
    parser.add_argument('--activation_command', type=str, default=None)
    parser.add_argument('--no_conncomp_on_paintera_export', action='store_true')
    parser.add_argument('--merge_small_segments', action='store_true')

    args = parser.parse_args()
    results_folder = args.result_folder
    inputs_folder = args.input_folder
    paintera_env_name = args.paintera_env_name
    activation_command = args.activation_command
    merge_small_segments = args.merge_small_segments

    conncomp_on_paintera_export = True
    if os.path.exists(os.path.join(inputs_folder, 'settings.json')):
        with open(os.path.join(inputs_folder, 'settings.json')) as f:
            settings = json.load(f)
        conncomp_on_paintera_export = settings['conncomp_on_paintera_export']

    if args.no_conncomp_on_paintera_export:
        conncomp_on_paintera_export = False
    if conncomp_on_paintera_export:
        print('Connected components on paintera export enabled.')
    else:
        print('Connected components on paintera export disabled.')

    assert inputs_folder
    print('paintera environment call: {} {}'.format(activation_command, paintera_env_name))
    if results_folder is None:
        inputs_folder = os.path.normpath(inputs_folder)
        # results_folder = os.path.join(os.path.split(inputs_folder)[0], 'result_{}_'.format(getpass.getuser()) + os.path.split(inputs_folder)[1])
        results_folder = os.path.join('/scratch/emcf/segmentation_results/', os.path.split(inputs_folder)[1])
        print('Writing results to {}'.format(results_folder))

    if not os.path.exists(results_folder):
        os.mkdir(results_folder)

    # TODO don't hard-code this, but also take as input args
    raw_filepath = os.path.join(
        inputs_folder,
        'raw.h5'
    )
    mem_filepath = os.path.join(
        inputs_folder,
        'mem.h5'
    )
    sv_filepath = os.path.join(
        inputs_folder,
        'sv.h5'
    )

    pm_workflow(
        results_folder=results_folder,
        raw_filepath=raw_filepath,
        mem_pred_filepath=mem_filepath,
        supervoxel_filepath=sv_filepath,
        mem_pred_channel=None,
        auto_crop_center=True,
        annotation_shape=(256, 256, 256),  # TODO: Retrieve from data size in sv.h5
        paintera_env_name=paintera_env_name,
        activation_command=activation_command,
        export_binary=True,
        conncomp_on_paintera_export=conncomp_on_paintera_export,
        merge_small_segments=merge_small_segments,
        verbose=True
    )
