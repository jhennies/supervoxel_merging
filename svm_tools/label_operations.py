
import numpy as np
import sys


def relabel_consecutive(map, sort_by_size=False):

    map = map + 1

    if sort_by_size:
        labels, counts = np.unique(map, return_counts=True)
        labels = labels[np.argsort(counts)[::-1]].tolist()
    else:
        labels = np.unique(map).tolist()
    relabel_dict = dict(zip(labels, range(len(labels))))

    # Perform the mapping
    c = 0
    map = map.astype('float32')
    for label, segment in relabel_dict.items():
        sys.stdout.write('\r' + 'Relabelling: {} %'.format(int(100 * float(c + 1) / float(len(relabel_dict)))))
        # print('label {} -> segment {}'.format(label, segment))
        map[map == label] = -segment
        c += 1
    map = (-map).astype('float32')

    return map

