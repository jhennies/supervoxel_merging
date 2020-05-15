
import numpy as np
import os
import sys
from h5py import File
import json
from vigra.analysis import labelMultiArray
from shutil import rmtree

from elf.io import open_file
import elf.segmentation.multicut as mc
import elf.segmentation.features as feats

from subprocess import call, run, DEVNULL
import multiprocessing as mp

import napari

from svm_tools.paintera_merge import convert_pre_merged_labels_to_assignments
from svm_tools.label_operations import relabel_consecutive
from svm_tools.volume_operations import crop_center

from svm_tools.position_generation_h5 import build_equally_spaced_volume_list


def _load_data(
        filepath,
        shape,
        auto_crop_center,
        channel=None,
        normalize=False,
        verbose=False,
        relabel=False,
        cache_folder=None
):
    with open_file(filepath, 'r') as f:
        data = f['data'][:]

    if cache_folder is None:
        name = os.path.splitext(filepath)[0]
    else:
        name = os.path.join(
            cache_folder,
            os.path.splitext(os.path.split(filepath)[1])[0]
        )
    if verbose:
        print(name)

    relabel_filepath = name + '_rl.h5'
    crop_filepath = name + '_crop_center{}.h5'.format('_'.join([str(x) for x in shape]))
    channel_filepath = name + '_ch{}.h5'.format(channel)

    if relabel and os.path.exists(relabel_filepath):
        if verbose:
            print('Loading relabeled data ...')
        with File(relabel_filepath, mode='r') as f:
            data = f['data'][:]
        filepath = relabel_filepath

    else:

        if auto_crop_center and os.path.exists(crop_filepath):
            if verbose:
                print('Loading cropped data ...')
            with File(crop_filepath, mode='r') as f:
                data = f['data'][:]
            filepath = crop_filepath

        else:

            if channel is not None:
                data = data[..., channel].squeeze()
                with File(channel_filepath, mode='w') as f:
                    f.create_dataset('data', data=data, compression='gzip')
                filepath = channel_filepath

            if auto_crop_center:
                if data.shape != shape:
                    data = crop_center(data, shape)

                with File(crop_filepath, mode='w') as f:
                    f.create_dataset('data', data=data, compression='gzip')
                filepath = crop_filepath

        if normalize:
            data = data.astype('float32')
            data /= 255

        if relabel:
            data = relabel_consecutive(data).astype('uint32')
            with File(relabel_filepath, mode='w') as f:
                f.create_dataset('data', data=data, compression='gzip')
            filepath = relabel_filepath

    if verbose:
        print('data.shape = {}'.format(data.shape))

    if channel:
        return data, filepath, channel_filepath
    else:
        return data, filepath


def _write_data(
        filepath,
        data,
        verbose=False
):
    if verbose:
        print('Writing to {}'.format(filepath))

    with File(filepath, 'w') as f:
        f.create_dataset('data', data=data, compression='gzip')


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


def _query_increase_decrease_value(question):

    valid = {"+": 'increase', "-": 'decrease'}
    prompt = " [+/-/float]"

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if choice in valid:
            return valid[choice]
        else:
            try:
                value = float(choice)
                return value
            except ValueError:
                sys.stdout.write("Please respond with '+' or '-' "
                                 "or a value.\n")


def _query_commands():

    valid = dict(
        update='update',
        u='update',
        exit='exit',
        q='exit',
        editor='editor',
        e='editor'
    )

    prompt = ' ?> '

    print('\nUse the command line for following commands:')
    print('            exit / q      -> finish assignments and export')
    print('            update / u    -> updates Napari display')
    print('            editor / e    -> re-opens editor')

    while True:
        sys.stdout.write(prompt)
        choice = input().lower()
        if choice in valid:
            return valid[choice]
        else:
            sys.stdout.write('No valid command')


def prepare_for_paintera(paintera_env_name, filepath, target_filepath,
                         activation_command='source activate', shell='/bin/bash',
                         src_name='data', tgt_name='data', verbose=False):

    if verbose:
        console_output = None
    else:
        console_output = DEVNULL
    if paintera_env_name is not None:
        activate = '{act} {paintera_env}\n'.format(act=activation_command, paintera_env=paintera_env_name)
        return call([
            '{act}'
            'paintera-convert to-paintera '
            '--container {src} --dataset {src_name} --output-container {tgt} --target-dataset {tgt_name}'.format(
                act=activate,
                src=filepath, src_name=src_name,
                tgt=target_filepath, tgt_name=tgt_name
            )
        ], shell=True, executable=shell, stdout=console_output, stderr=console_output)
    else:
        return run([
            'bash --login -c '
            '"paintera-convert to-paintera '
            '--container {src} --dataset {src_name} --output-container {tgt} --target-dataset {tgt_name}"'.format(
                src=filepath, src_name=src_name,
                tgt=target_filepath, tgt_name=tgt_name
            )
        ], shell=True, executable=shell, stdout=console_output, stderr=console_output)


def export_from_paintera(paintera_env_name, filepath, target_filepath,
                         activation_command='source activate', shell='/bin/bash',
                         src_name='data', tgt_name='data', verbose=False):
    if verbose:
        console_output = None
    else:
        console_output = DEVNULL
    if paintera_env_name is not None:
        activate = '{act} {paintera_env}\n'.format(act=activation_command, paintera_env=paintera_env_name)
        return call([
            '{act}'
            'paintera-convert to-scalar '
            '--consider-fragment-segment-assignment -i {fp} -I {src_name} -o {target_fp} -O {tgt_name}'.format(
                act=activate,
                fp=filepath,
                target_fp=target_filepath,
                src_name=src_name,
                tgt_name=tgt_name
            )
        ], shell=True, executable=shell, stdout=console_output, stderr=console_output)
    else:
        return run([
            'bash --login -c '
            '"paintera-convert to-scalar '
            '--consider-fragment-segment-assignment -i {fp} -I {src_name} -o {target_fp} -O {tgt_name}"'.format(
                fp=filepath,
                target_fp=target_filepath,
                src_name=src_name,
                tgt_name=tgt_name
            )
        ], shell=True, executable=shell, stdout=console_output, stderr=console_output)


def open_paintera(paintera_env_name, project_folder,
                  activation_command='source activate', shell='/bin/bash', verbose=False):
    if verbose:
        console_output = None
    else:
        console_output = DEVNULL
    if paintera_env_name is not None:
        activate = '{act} {paintera_env}\n'.format(act=activation_command, paintera_env=paintera_env_name)
        return call([
            '{act}'
            'paintera {folder}'.format(
                act=activate,
                folder=project_folder
            )
        ], shell=True, executable=shell, stdout=console_output, stderr=console_output)
    else:
        return call([
            'bash --login -c '
            '"paintera {folder}"'.format(
                folder=project_folder
            )
        ], shell=True, executable=shell, stdout=console_output, stderr=console_output)


def open_napari(data):

    with napari.gui_qt():
        viewer = napari.Viewer()
        for item in data:
            if item['type'] == 'label':
                viewer.add_labels(item['data'], name=item['name'], visible=item['visible'])
            elif item['type'] == 'raw':
                viewer.add_image(item['data'], name=item['name'], visible=item['visible'])

    return None


def _open_editor(filepath):

    run([
        'bash --login -c '
        '"gedit {fp}"'.format(fp=filepath)
    ], shell=True)


def supervoxel_merging(mem, sv, beta=0.5, verbose=False):

    rag = feats.compute_rag(sv)
    costs = feats.compute_boundary_features(rag, mem)[:, 0]

    edge_sizes = feats.compute_boundary_mean_and_length(rag, mem)[:, 1]
    costs = mc.transform_probabilities_to_costs(costs, edge_sizes=edge_sizes, beta=beta)

    node_labels = mc.multicut_kernighan_lin(rag, costs)
    segmentation = feats.project_node_labels_to_pixels(rag, node_labels)

    return segmentation


def multicut_module(
        seg_filepath,
        raw, mem, sv,
        verbose=False
):
    if not os.path.exists(seg_filepath):

        user_happy = False
        beta = 0.5
        while not user_happy:
            print(sv.shape)
            # 1. Run Multicut
            seg = supervoxel_merging(mem, sv, beta=beta, verbose=verbose)

            # 2. Show results in Napari
            print('\nShowing Multicut result for beta = {}'.format(beta))
            print('Decide whether you are happy or there should be more or less merges and close Napari.')
            to_show = [dict(type='raw', name='raw', data=raw, visible=True)]
            if mem is not None:
                to_show.append(dict(type='raw', name='mem', data=mem, visible=True))
            to_show.append(dict(type='label', name='sv', data=sv, visible=True))
            to_show.append(dict(type='label', name='seg', data=seg, visible=True))
            open_napari(to_show)

            # 3. Ask user if results are good
            #    If not, go back to run multicut (1), else continue
            user_happy = query_yes_no('Happy with the result?', default='no')
            if not user_happy:
                change_beta = _query_increase_decrease_value(
                    'Less merges (increase beta) or more merges (decrease beta)?')
                if type(change_beta) == str:
                    if change_beta == 'increase':
                        beta += 0.1
                        if beta > 1:
                            beta = 1
                    elif change_beta == 'decrease':
                        beta -= 0.1
                        if beta < 0:
                            beta = 0
                    else:
                        raise ValueError
                elif type(change_beta) == float:
                    beta = change_beta
                    if beta > 1:
                        beta = 1
                    if beta < 0:
                        beta = 0
                else:
                    raise ValueError

        _write_data(seg_filepath, seg)

    else:
        print('Segmentation exists')


def paintera_merging_module(
        results_folder,
        paintera_proj_path,
        activation_command,
        paintera_env_name,
        shell,
        seg_filepath, seg_name,
        full_raw_filepath, raw_name,
        supervoxel_filepath, sv_name,
        mem_pred_filepath, mem_name,
        verbose
):

    supervoxel_proj_path = os.path.join(results_folder, 'data.n5')

    if not os.path.exists(os.path.join(results_folder, 'data.n5')):
        if verbose:
            print('Preparing raw ...')
        print('\n>>> SHELL >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')
        if prepare_for_paintera(paintera_env_name, full_raw_filepath, os.path.join(results_folder, 'data.n5'),
                                activation_command, shell, verbose=verbose, src_name=raw_name, tgt_name='raw'):
            pass
            # raise RuntimeError
        print('\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n')

        if mem_pred_filepath is not None:
            if verbose:
                print('Preparing membrane prediction ...')
            print('\n>>> SHELL >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')
            if prepare_for_paintera(paintera_env_name, mem_pred_filepath, os.path.join(results_folder, 'data.n5'),
                                    activation_command, shell, verbose=verbose, src_name=mem_name, tgt_name='mem'):
                pass
                # raise RuntimeError
            print('\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n')

        if verbose:
            print('Preparing supervoxels ...')
        print('\n>>> SHELL >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')
        if prepare_for_paintera(paintera_env_name, supervoxel_filepath, supervoxel_proj_path,
                                activation_command, shell, verbose=verbose, src_name=sv_name, tgt_name='sv'):
            pass
            # raise RuntimeError
        print('\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n')

        if verbose:
            print('Assigning pre-merged segmentation to supervoxels')
        convert_pre_merged_labels_to_assignments(
            supervoxel_filepath, seg_filepath, paintera_proj_path, sv_name='data', merged_name=seg_name
        )

    if not os.path.exists(os.path.join(paintera_proj_path, 'attributes.json')):
        # 5. Ask user to create paintera project and close paintera again
        print('\n\nOpening paintera...\n')
        print('Set up the Paintera project by loading the following data from')
        print(os.path.join(results_folder, 'data.n5'))
        print('Dataset names:')
        print('1. Raw data (as type raw):                "raw"')
        if mem_pred_filepath is not None:
            print('2. Membrane prediction (as type raw):     "mem"')
            print('3. Supervoxels (as type labels):          "sv"')
        else:
            print('2. Supervoxels (as type labels):          "sv"')

    else:
        print('\nPaintera project exists ...')

    print('\nProof-read the segmentation as desired; save, commit changes and close Paintera when done.')
    print('\nNote: DO NOT FORGET TO COMMIT CHANGES BEFORE CLOSING!')

    print('\n>>> SHELL >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')
    if open_paintera(paintera_env_name, paintera_proj_path, activation_command, shell, verbose=verbose):
        raise RuntimeError
    print('\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n')

    # 8. Convert results to h5 and do the assignments
    #    Generate terminal commands:
    #    > conda activate paintera_env
    #    > paintera-convert to-scalar --consider-fragment-segment-assignment ...
    print('Exporting from paintera ...')
    if verbose:
        print(supervoxel_proj_path)
    print('\n>>> SHELL >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')
    if export_from_paintera(paintera_env_name, supervoxel_proj_path, os.path.join(results_folder, 'exported_seg.h5'),
                            activation_command, shell, src_name='sv', tgt_name='data',
                            verbose=verbose):
        pass
        # raise RuntimeError
    print('\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n')
    assert os.path.exists(os.path.join(results_folder, 'exported_seg.h5'))


def paintera_merging_module2(
        tmp_dir,
        results_folder,
        paintera_lock_file,
        paintera_proj_path,
        activation_command,
        paintera_env_name,
        shell,
        seg_filepath,
        full_raw_filepath,
        supervoxel_filepath,
        supervoxel_proj_path,
        mem_pred_filepath,
        result_dtype,
        conncomp_on_paintera_export,
        export_filepath=None,
        export_name=None,
        verbose=False
):
    if verbose:
        print('Preparing raw ...')

    raw_n5 = os.path.join(tmp_dir, 'raw.n5')
    if os.path.exists(raw_n5):
        if os.path.exists(os.path.join(raw_n5, 'attributes.json')):
            rmtree(raw_n5)
        else:
            print('raw.n5 exists but is not n5')
            raise RuntimeError
    # if not os.path.exists(os.path.join(results_folder, 'raw.n5')):
    if True:
        print('\n>>> SHELL >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')
        prepare_for_paintera(paintera_env_name, full_raw_filepath, os.path.join(tmp_dir, 'raw.n5'),
                             activation_command, shell)
        print('\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n')
    # if mem_pred_filepath is not None:
    #     if verbose:
    #         print('Preparing membrane prediction ...')
    #     if not os.path.exists(os.path.join(results_folder, 'mem.n5')):
    #         print('\n>>> SHELL >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')
    #         prepare_for_paintera(paintera_env_name, mem_pred_filepath, os.path.join(results_folder, 'mem.n5'),
    #                               activation_command, shell)
    #         print('\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n')
    if verbose:
        print('Preparing supervoxels ...')
    sv_n5 = os.path.join(tmp_dir, 'sv.n5')
    if os.path.exists(sv_n5):
        if os.path.exists(os.path.join(sv_n5, 'attributes.json')):
            rmtree(sv_n5)
        else:
            print('sv.n5 exists but is not n5')
            raise RuntimeError
    # if not os.path.exists(os.path.join(results_folder, 'sv.n5')):
    if True:
        print('\n>>> SHELL >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')
        prepare_for_paintera(paintera_env_name, supervoxel_filepath, supervoxel_proj_path,
                             activation_command, shell)
        print('\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n')

    if not os.path.exists(os.path.join(paintera_proj_path, 'attributes.json')):
        # 5. Ask user to create paintera project and close paintera again
        print('\n\nOpening paintera...\n')
        print('Set up the Paintera project by loading the following files from')
        print(tmp_dir)
        print('1. Raw data (as type raw):                "raw.n5"')
        if mem_pred_filepath is not None:
            print('2. Membrane prediction (as type raw):     "mem.n5"')
            print('3. Supervoxels (as type labels):          "sv.n5"')
        else:
            print('2. Supervoxels (as type labels):          "sv.n5"')

        print('\nIt is possible to change settings at this step, for example, '
              'it is best to already switch off 3D rendering.')
        print('\nNote: DO NOT YET PERFORM ANY MERGES!')

        print('\nSave and close Paintera when done.')

        print('\n>>> SHELL >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')
        open_paintera(paintera_env_name, paintera_proj_path, activation_command, shell)
        print('\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n')
    else:
        print('\nPaintera project exists ...')

    # TODO why is there an if True here?
    # if not os.path.exists(paintera_lock_file):
    if True:
        print('\nIntegrating pre-merged segmentation into Paintera project ...')
        # 6. Integrate Multicut result to the paintera project
        convert_pre_merged_labels_to_assignments(
            supervoxel_filepath, seg_filepath, paintera_proj_path
        )
        open(paintera_lock_file, 'a').close()
    else:
        print('\nPaintera project already locked ...')

    # 7. Ask user to open paintera again, to do the necessary annotations and then close paintera
    print('\n\nOpening paintera...\n')
    print('Perform the necessary corrections, then save and close Paintera.')
    print('Consider committing to backend (CTRL + C) before starting to annotate to make Paintera run more fluently.')
    print('\n>>> SHELL >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')
    open_paintera(paintera_env_name, paintera_proj_path, activation_command, shell)
    print('\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n')

    # 8. Convert results to h5 and do the assignments
    #    Generate terminal commands:
    #    > conda activate paintera_env
    #    > paintera-convert to-scalar --consider-fragment-segment-assignment ...
    print('Exporting from paintera ...')
    if export_filepath is None:
        export_filepath = os.path.join(results_folder, '{}.h5'.format(export_name))
    export_from_paintera(paintera_env_name, supervoxel_proj_path, export_filepath,
                         activation_command, shell)

    if conncomp_on_paintera_export or result_dtype is not None:
        with open_file(export_filepath, mode='r') as f:
            exp_seg = f['data'][:]
        if conncomp_on_paintera_export:
            # Computing connected components
            exp_seg = labelMultiArray(exp_seg.astype('float32'))
        exp_seg = relabel_consecutive(exp_seg, sort_by_size=True)

        if result_dtype is not None:
            exp_seg = exp_seg.astype(result_dtype)

        with File(export_filepath, mode='w') as f:
            f.create_dataset('data', data=exp_seg, compression='gzip')


def organelle_assignment_module(
        results_folder,
        organelle_assignments_filepath,
        export_name,
        mem_pred_filepath,
        raw,
        mem,
        conncomp_on_paintera_export,
        result_dtype,
        export_binary,
        verbose
):
    print('\nNapari and editor started in sub-processes.')
    print('\nFill the text file with assignments')

    with open_file(os.path.join(results_folder, 'exported_seg.h5'), mode='r') as f:
        exp_seg = f['data'][:]
    if conncomp_on_paintera_export:
        # Computing connected components
        exp_seg = labelMultiArray(exp_seg.astype('float32'))
    exp_seg = relabel_consecutive(exp_seg, sort_by_size=True)

    if not os.path.exists(organelle_assignments_filepath):
        with open(organelle_assignments_filepath, mode='w') as f:
            json.dump(dict(CYTO=dict(labels=[0], type='single')), f)

    all_ids = list(np.unique(exp_seg))

    def _generate_organelle_maps():

        try:
            # TODO this should be wrapped in a try/except in case of invalid json syntax
            # and then be caught to tell user to correct it
            # get the current organelle assignments from the text file
            with open(organelle_assignments_filepath, mode='r') as f:
                assignments = json.load(f)

            maps = {}
            assigned = []
            for organelle, assignment in assignments.items():
                print('found organelle: {}'.format(organelle))
                maps[organelle] = np.zeros(exp_seg.shape, dtype=exp_seg.dtype)
                val = 1
                for idx in assignment['labels']:
                    maps[organelle][exp_seg == idx] = val
                    if assignment['type'] == 'multi':
                        val += 1
                    assigned.append(idx)

            unassigned = np.setdiff1d(all_ids, assigned)
            maps['MISC'] = np.zeros(exp_seg.shape, dtype=exp_seg.dtype)
            val = 1
            for idx in unassigned:
                maps['MISC'][exp_seg == idx] = val
                val += 1

            map_names = sorted(maps.keys())
            maps['SEMANTICS'] = np.zeros(exp_seg.shape, dtype=exp_seg.dtype)
            for map_idx, map_name in enumerate(map_names):
                maps['SEMANTICS'][maps[map_name] > 0] = map_idx

            return maps
        except:
            print('Invalid json syntax!!! Fix the json file, save and update Napari again!')
            return {}

    def _print_help():
        # I don't think we need explicit quit command any more
        # print('            exit / q      -> finish assignments and export')
        print('            update / u    -> updates Napari display')
        print('            editor / e    -> re-opens editor')

    # start the editor in a sub-process
    editor_p = mp.Process(target=_open_editor, args=(organelle_assignments_filepath,))
    editor_p.start()

    with napari.gui_qt():
        viewer = napari.Viewer()
        # add the initiail (static) layers
        viewer.add_image(raw, name='raw')
        if mem_pred_filepath is not None:
            viewer.add_image(mem, name='mem', visible=False)
        viewer.add_labels(exp_seg, name='from Paintera', visible=False)

        # add the initial organelle maps
        organelle_maps = _generate_organelle_maps()
        for name, data in organelle_maps.items():
            is_visible = name == 'MISC'
            viewer.add_labels(data, name=name, visible=is_visible)

        _print_help()

        # I don't think this is necessary any more
        # @viewer.bind_key('q')
        # def quit(viewer):
        #     pass

        @viewer.bind_key('h')
        def help(viewer):
            _print_help()

        @viewer.bind_key('u')
        def update(viewer):
            print("Updating napari layers from organelle assignments ...")
            new_organelle_maps = _generate_organelle_maps()

            # iterate over the organelle maps, if we have it in the layers already, update the layer,
            # otherwise add a new layer
            # TODO this does not catch the case where a category is removed yet (the layer will persist)
            # this should also be caught and the layer be removed
            layers = viewer.layers
            for name, data in new_organelle_maps.items():
                is_visible = name == 'MISC'
                # if name in layers:
                try:
                    # This raises a key error if the layer does not exist
                    # FIXME is there a solution like 'if name in layers: ...' that does not error out?
                    name in layers
                    layers[name].data = data
                except KeyError:
                    viewer.add_labels(data, name=name, visible=is_visible)
            print("... done")

        @viewer.bind_key('e')
        def editor(viewer):
            nonlocal editor_p
            editor_p.terminate()
            editor_p.join()
            editor_p = mp.Process(target=_open_editor, args=(organelle_assignments_filepath,))
            editor_p.start()

    # 10. Export organelle maps
    print('Exporting organelle maps ...')
    organelle_maps = _generate_organelle_maps()
    for map_name, map in organelle_maps.items():
        if not os.path.exists(os.path.join(os.path.join(results_folder, 'results'))):
            os.mkdir(os.path.join(os.path.join(results_folder, 'results')))
        # Export labeled result
        organelle_filepath = os.path.join(
            results_folder,
            'results',
            export_name + '_{}.h5'.format(map_name)
        )
        _write_data(organelle_filepath, map.astype(result_dtype), verbose=verbose)

        if export_binary:
            # Export binary result
            organelle_filepath = os.path.join(
                results_folder,
                'results',
                export_name + '_{}_bin.h5'.format(map_name)
            )
            map = (1 - (1 - map.astype('float32') / map.max()).astype('uint8')) * 255
            _write_data(organelle_filepath, map, verbose=verbose)


def data_loading_module(
        results_folder,
        supervoxel_filepath,
        raw_filepath,
        mem_pred_filepath,
        mem_pred_channel,
        annotation_shape,
        auto_crop_center,
        verbose
):
    seg_filepath = os.path.join(
        results_folder,
        os.path.splitext(os.path.split(supervoxel_filepath)[1])[0] + '_seg.h5'
    )

    # Load raw, mem prediction, and supervoxels
    full_raw_filepath = raw_filepath
    raw, raw_filepath = _load_data(raw_filepath, annotation_shape, auto_crop_center,
                                   verbose=verbose, cache_folder=results_folder)
    if mem_pred_filepath is not None:
        if mem_pred_channel is not None:
            mem, mem_pred_filepath, mem_pred_channel_fp = _load_data(mem_pred_filepath, annotation_shape,
                                                                     auto_crop_center,
                                                                     normalize=True, channel=mem_pred_channel,
                                                                     verbose=verbose,
                                                                     cache_folder=results_folder)
        else:
            mem, mem_pred_channel_fp = _load_data(mem_pred_filepath, annotation_shape, auto_crop_center,
                                                  normalize=True, channel=mem_pred_channel, verbose=verbose,
                                                  cache_folder=results_folder)
    else:
        mem = None
        mem_pred_channel_fp = None
    sv, supervoxel_filepath = _load_data(supervoxel_filepath, annotation_shape,
                                         auto_crop_center, verbose=verbose, relabel=True, cache_folder=results_folder)

    return(seg_filepath,
           full_raw_filepath,
           raw_filepath,
           mem_pred_filepath,
           mem_pred_channel_fp,
           supervoxel_filepath,
           raw, mem, sv)


def proof_reading_workflow(
        results_folder,
        raw_filepath,
        supervoxel_folder,
        seg_filepath=None,
        seg_folder=None,
        seg_filename_pattern='result_lmc_{z}_{y}_{x}.h5',
        # mem_pred_filepath,
        sv_filename_pattern='{z}_{y}_{x}.h5',
        # mem_pred_channel=None,
        # auto_crop_center=False,
        # annotation_shape=None,
        paintera_env_name='paintera_env',
        activation_command='source',
        shell='/bin/bash',
        result_dtype='uint16',
        # export_binary=False,
        conncomp_on_paintera_export=True,
        # pre_segmentation_filepath=None,
        verbose=False
):

    if not os.path.exists(results_folder):
        os.mkdir(results_folder)

    assert seg_filepath is not None or seg_folder is not None, 'Either seg_filepath or seg_folder has to be given!'
    if seg_filepath is not None and seg_folder is not None:
        print('Warning: Both seg_filepath and seg_folder given. Using seg_filepath = {}'.format(seg_filepath))

    with File(raw_filepath, mode='r') as f:
        full_shape = f['data'].shape

    if verbose:
        print('Full dataset shape = {}'.format(full_shape))

    n, position_list = build_equally_spaced_volume_list(
        full_shape,
        target_shape=(512, 512, 512),
        overlap=(256, 256, 256),
        overshoot=True
    )

    print('Total number of cubes = {}'.format(n))
    if verbose:
        print(position_list)

    # Make a temp directory to store intermediate files
    tmp_dir = os.path.join(results_folder, 'tmp')
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)

    # __________________________________________________________________________________________________________________
    # Iterate over full dataset
    for pidx, position in enumerate(position_list):

        x = position[1][2].start
        y = position[1][1].start
        z = position[1][0].start
        export_filepath = os.path.join(results_folder, seg_filename_pattern.format(x=x, y=y, z=z))

        if os.path.exists(export_filepath):
            print('Cube {} / {} exists: {}'.format(pidx + 1, n, seg_filename_pattern.format(x=x, y=y, z=z)))
        else:

            print('Proof reading of cube {} / {}: {}'.format(pidx + 1, n, seg_filename_pattern.format(x=x, y=y, z=z)))
            if verbose:
                print(position)

            # ______________________________________________________________________________________________________________
            # Load data
            with File(raw_filepath, mode='r') as f:
                raw = f['data'][position[1]].astype('uint8')
                if verbose:
                    print('Raw shape = {}'.format(raw.shape))
                    print('Raw dtype = {}'.format(raw.dtype))

            if seg_filepath is not None:
                if verbose:
                    print('Opening {}'.format(seg_filepath))
                with File(seg_filepath, mode='r') as f:
                    seg = f['data'][position[1]]
            else:
                filepath = os.path.join(seg_folder, seg_filename_pattern.format(x=x, y=y, z=z))
                if verbose:
                    print('Opening {}'.format(filepath))
                with File(filepath, mode='r') as f:
                    seg = f['data'][:]
                if verbose:
                    print('seg shape = {}'.format(seg.shape))

            # Make a ROI map
            roi_map = np.ones(raw.shape, dtype=bool)
            roi_map[128: -128, 128: -128, 128: -128] = 0

            # Visualize in napari
            data = [
                dict(type='raw', data=raw, name='raw', visible=True),
                dict(type='label', data=seg, name='seg', visible=True),
                dict(type='label', data=roi_map, name='roi', visible=True)
            ]
            open_napari(data)

            if not query_yes_no('Needs corrections?'):
                continue

            # print(np.unique(raw))
            # print(raw.shape)
            # print(raw.dtype)
            raw[roi_map] = raw[roi_map] * 0.7
            raw = raw.astype('uint8')
            if verbose:
                print(raw.shape)
                print(raw.dtype)

            tmp_seg_filepath = os.path.join(tmp_dir, 'seg.h5')
            with File(tmp_seg_filepath, mode='w') as f:
                f.create_dataset('data', data=seg, compression='gzip')
            tmp_raw_filepath = os.path.join(tmp_dir, 'raw.h5')
            with File(tmp_raw_filepath, mode='w') as f:
                f.create_dataset('data', data=raw, compression='gzip')

            # ______________________________________________________________________________________________________________
            # Corrections with paintera
            paintera_merging_module2(
                tmp_dir, results_folder,
                paintera_lock_file=os.path.join(tmp_dir, '.paintera_lock'),
                paintera_proj_path=os.path.join(results_folder, 'paintera_proj'),
                activation_command=activation_command,
                paintera_env_name=paintera_env_name,
                shell=shell,
                seg_filepath=tmp_seg_filepath,
                full_raw_filepath=tmp_raw_filepath,
                supervoxel_filepath=os.path.join(supervoxel_folder, sv_filename_pattern.format(x=x, y=y, z=z)),
                supervoxel_proj_path=os.path.join(tmp_dir, 'sv.n5'),
                mem_pred_filepath=None,
                result_dtype=result_dtype,
                conncomp_on_paintera_export=conncomp_on_paintera_export,
                export_filepath=export_filepath,
                verbose=verbose
            )


def pm_workflow(
        results_folder,
        raw_filepath,
        mem_pred_filepath,
        supervoxel_filepath,
        raw_name='data',
        mem_name='data',
        sv_name='data',
        mem_pred_channel=None,
        auto_crop_center=False,
        annotation_shape=None,
        paintera_env_name='paintera_env',
        activation_command='source',
        shell='/bin/bash',
        result_dtype='uint16',
        export_binary=False,
        conncomp_on_paintera_export=True,
        pre_segmentation_filepath=None,
        verbose=False
):
    """

    :param results_folder:
    :param raw_filepath:
    :param mem_pred_filepath:
    :param supervoxel_filepath:
    :param mem_pred_channel:
    :param auto_crop_center:
    :param annotation_shape:
    :param paintera_env_name:
    :param activation_command:
    :param shell:
    :param result_dtype:
    :param export_binary:
    :param conncomp_on_paintera_export:
    :param pre_segmentation_filepath: Filepath to volume with same size as supervoxels.
            If this is given, no multicut is computed and this volume used instead
    :param verbose:
    :return:
    """

    paintera_proj_path = os.path.join(results_folder, 'paintera_proj')
    organelle_assignments_filepath = os.path.join(results_folder, 'organelle_assignments.json')

    if not os.path.exists(results_folder):
        os.mkdir(results_folder)

    if annotation_shape is None:
        with open_file(supervoxel_filepath, mode='r') as f:
            annotation_shape = f['data'].shape
        print('Annotation shape set to {}'.format(annotation_shape))

    raw_filename = os.path.split(raw_filepath)[1]

    # __________________________________________________________________________________________________________________
    # Supervoxel merging with Multicut

    (
        seg_filepath,
        full_raw_filepath,
        raw_filepath,
        mem_pred_filepath,
        mem_pred_channel_fp,
        supervoxel_filepath,
        raw, mem, sv
    ) = data_loading_module(
        results_folder,
        supervoxel_filepath,
        raw_filepath,
        mem_pred_filepath,
        mem_pred_channel,
        annotation_shape,
        auto_crop_center,
        verbose
    )

    if pre_segmentation_filepath is None:

        multicut_module(
            seg_filepath,
            raw, mem, sv,
            verbose=verbose
        )

    # __________________________________________________________________________________________________________________
    # Supervoxel merging with Paintera

    # 4. Convert raw data, membrane prediction and supervoxels to n5
    #    Generate terminal commands:
    #    > conda activate paintera_env
    #    > paintera-convert to-paintera ... (for supervoxels, mem pred, and raw, each)
    #    > conda deactivate

    # 7. Ask user to open paintera again, to do the necessary annotations and then close paintera

    # 8. Convert results to h5 and do the assignments
    #    Generate terminal commands:
    #    > conda activate paintera_env
    #    > paintera-convert to-scalar --consider-fragment-segment-assignment ...

    paintera_merging_module(
        results_folder,
        paintera_proj_path,
        activation_command,
        paintera_env_name,
        shell,
        seg_filepath, 'data',
        full_raw_filepath, raw_name,
        supervoxel_filepath, sv_name,
        mem_pred_channel_fp, mem_name,
        verbose
    )

    # __________________________________________________________________________________________________________________
    # 9. Assignment of organelles
    #    Open Napari in subprocess
    #    Open Editor in subprocess
    #    Enable commands in main process to re-open and update napari

    organelle_assignment_module(
        results_folder,
        organelle_assignments_filepath,
        os.path.splitext(raw_filename)[0],
        mem_pred_filepath,
        raw,
        mem,
        conncomp_on_paintera_export,
        result_dtype,
        export_binary,
        verbose
    )


if __name__ == '__main__':

    results_path = '/data/tmp/pr_test'

    proof_reading_workflow(
        results_path,
        raw_filepath='/data/phd_project/image_analysis/psp_full_experiments/'
                     'psp_200107_00_ds_20141002_hela_wt_xyz8nm_as_multiple_scales/step0_datasets/'
                     'psp0_200108_01_amst_align_psp0_200108_00/crop_inv_pad.h5',
        # seg_folder='/data/phd_project/image_analysis/psp_full_experiments/'
        #              'psp_200107_00_ds_20141002_hela_wt_xyz8nm_as_multiple_scales/step4_one_organelle_mc/'
        #              'psp4_200114_04_train_3cubes_er_beta_0.85/run_200119_00_er_full_dataset',
        seg_filepath='/data/phd_project/image_analysis/psp_full_experiments/'
                     'psp_200107_00_ds_20141002_hela_wt_xyz8nm_as_multiple_scales/step4_one_organelle_mc/'
                     'psp4_200114_04_train_3cubes_er_beta_0.85/run_200119_00_er_full_dataset_stiches/'
                     'stitch_binarized_er_curated.h5',
        supervoxel_folder='/data/phd_project/image_analysis/psp_full_experiments/'
                          'psp_200107_00_ds_20141002_hela_wt_xyz8nm_as_multiple_scales/step2_supervoxels/'
                          'psp2_200117_00_full_ds_pos_gen/20141002_hela_wt_xyz8nm_as/',
        paintera_env_name='paintera_env_new',
        activation_command='source /home/hennies/miniconda3/bin/activate',
        conncomp_on_paintera_export=False,
        verbose=False
    )

    # results_path = '/g/schwab/hennies/tmp/pm_test'
    # raw_fp = os.path.join(
    #     '/g/schwab/hennies/phd_project/image_analysis/psp_full_experiments/psp_191120_00_ds_hela_switch_to_elf/step0_datasets/g_segp_190508_00_make_test_datasets',
    #     'x1374_y991_z799.h5'
    # )
    # mem_fp = os.path.join(
    #     '/g/schwab/hennies/phd_project/image_analysis/psp_full_experiments/psp_191120_00_ds_hela_switch_to_elf/step1_membrane_predictions/g_psp1_190919_00_run_unet3d_190522_00_on_test_datasets',
    #     'x1374_y991_z799.h5'
    # )
    # sv_fp = os.path.join(
    #     '/g/schwab/hennies/phd_project/image_analysis/psp_full_experiments/psp_191120_00_ds_hela_switch_to_elf/step2_supervoxels/g_psp2_190919_00_wsdt_with_multiple_threshs_on_test_data/128to-128_128to-128_128to-128_tophat_ws_on_dt',
    #     'x1374_y991_z799.h5'
    # )
    #
    # pm_workflow(
    #     results_path,
    #     raw_fp,
    #     mem_fp,
    #     sv_fp,
    #     mem_pred_channel=2,
    #     auto_crop_center=True,
    #     annotation_shape=(256, 256, 256),
    #     paintera_env_name='paintera_env_new',
    #     result_dtype='uint16',
    #     export_binary=True,
    #     show_mem_pred=True,
    #     pre_segmentation_filepath=None,
    #     verbose=True
    # )
