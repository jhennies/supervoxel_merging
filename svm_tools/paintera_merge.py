
import numpy as np
import os
from h5py import File
import json


def get_actions_from_label_maps(sv, merged_sv):

    merge_actions = []
    sv_max = sv.max()

    # Iterate over each object in the merged version
    for idx in np.unique(merged_sv):

        # print('IDX = {}'.format(idx))

        # Get the supervoxels from each object
        sv_ids = np.unique(sv[merged_sv == idx])

        # print('SV_IDS = {}'.format(sv_ids))

        # Generate merge action strings
        ref_idx = 0
        for count_idx, sv_idx in enumerate(sv_ids):

            if count_idx == 0:
                ref_idx = sv_idx

            else:
                merge_actions.append(
                    dict(
                        type='MERGE',
                        data=dict(
                            fromFragmentId=int(sv_idx),
                            intoFragmentId=int(ref_idx),
                            segmentId=int(idx + sv_max + 1)
                        )
                    )
                )

    return merge_actions


def write_actions_to_attributes(actions, paintera_proj_path):

    attributes_fp = os.path.join(paintera_proj_path, 'attributes.json')
    with open(attributes_fp, mode='r') as f:
        data = json.load(f)

    for sidx, source in enumerate(data['paintera']['sourceInfo']['sources']):
        if source['type'] == 'org.janelia.saalfeldlab.paintera.state.LabelSourceState':
            data['paintera']['sourceInfo']['sources'][sidx]['state']['assignment']['data']['actions'] = actions

    with open(attributes_fp, mode='w') as f:
        json.dump(data, f)


def convert_pre_merged_labels_to_paintera(sv_filepath, merged_filepath, paintera_proj_path):
    """
    This function integrates merged supervoxels into paintera.

    The workflow:
    1. Merge supervoxels with any method (e.g. Multicut) and save result as HDF5 (as label map)
    2. Create a paintera project and load the supervoxel map as label image
    3. Close the paintera project
    4. Run this function with following parameters:

    :param sv_filepath: Path to a h5 file containing the supervoxels
    :param merged_filepath: Path to a h5 file containing the segmentation
                Note: This segmentation must be obtained by MERGING supervoxels ONLY! Otherwise this script fails
    :param paintera_proj_path: The path to the paintera project
    
    5. Open the paintera project again. The supervoxels are now merged according to the merges performed in step 1 but
        can be split off again if necessary.

    """

    with File(merged_filepath, mode='r') as f:
        merged_sv = f['data'][:]

    with File(sv_filepath, mode='r') as f:
        sv = f['data'][:]

    acts = get_actions_from_label_maps(sv, merged_sv)
    write_actions_to_attributes(acts, paintera_proj_path)
