import os
import h5py
import numpy as np
import z5py


def get_assignments_from_labels(sv, merged_sv):
    # NOTE we use the indices of np.unique to get the associated segment ids
    # this assumes that sv and merged_sv are perfectly aligned, otherwise one would
    # need to assign based on max overlaps
    fragment_ids, indices = np.unique(sv, return_index=True)
    segment_ids = merged_sv.ravel()[indices]

    # fragment_ids and segment_ids need to be disjoint, so we offset the segment ids with
    # the max fragment id
    max_frag_id = int(fragment_ids.max())
    segment_ids += max_frag_id

    # paintera assignment format:
    # [[frag_id1, seg_id1], [frag_id2, seg_id1], ..., [frag_idN, seg_idM]]
    assignments = np.concatenate([fragment_ids[:, None], segment_ids[:, None]], axis=1).T
    return assignments


def write_assignments_to_paintera_format(assignments, data_path, paintera_prefix):
    with z5py.File(data_path) as f:
        g = f[paintera_prefix]
        ds = g.require_dataset('fragment-segment-assignment', shape=assignments.shape, compression='gzip',
                               chunks=assignments.shape, dtype=assignments.dtype)
        ds[:] = assignments


# NOTE for now we nned to weirdly hard-code the path to the data, because it is stored in the 'sv.n5' file
# in the top-dir of paintera_project_path.
# I would recommend to restructure this and have everything in one 'data.n5' container.
# And then pass this data-path and the paintera data prefix to this function
def convert_pre_merged_labels_to_assignments(sv_filepath, merged_filepath, paintera_proj_path,
                                             sv_name='sv', merged_name='data'):
    """ Accumulate node labels for superpixels and write out paintera fragment-segment-assignment format
    """
    with h5py.File(merged_filepath, mode='r') as f:
        merged_sv = f[merged_name][:]

    # with h5py.File(sv_filepath, mode='r') as f:
    #     sv = f[sv_name][:]
    # FIXME use the n5 here!
    with h5py.File(sv_filepath, mode='r') as f:
        sv = f[sv_name][:]

    assignments = get_assignments_from_labels(sv, merged_sv)
    data_path = os.path.join(os.path.split(paintera_proj_path)[0], 'data.n5')
    write_assignments_to_paintera_format(assignments, data_path, 'sv')
