#!/usr/bin/env python

from h5py import File
import numpy as np

# _______________________________________________________________________
# part 1: flatten function

feature_names = [u'clusters', u'jets', u'subjet1', u'subjet2', u'tracks']
# feature_names = [u'jets']

def flatten(ds):
    """
    Flattens a named numpy array so it can be used with pure numpy.
    Input: named array
    Output: numpy array

    Example;
    print(flatten(sample_jets[['pt', 'eta']]))
    """

    # we want to cast to a new type, all floats
    ar = np.asarray(ds)
    nms = ar.dtype.names
    ftype = [(n, float) for n in nms]

    # The flattening will put the new 'feature' dimension last
    flat = ar.astype(ftype, casting='safe').view((float, len(nms)))

     # so we roll to put the 'feature number' on axis 1
    return np.rollaxis(flat, -1, 1)

# ________________________________________________________________________
# Part 2: sub selections

def get_subselection(data, sub_selection):
    batch_size = data[0].shape[0]
    slices = []
    for dat, subsel in zip(data, sub_selection):
        # take a slice of the data, and then reshape it to flatten
        # everything past the first dimenstion
        subsel_dat = dat[:,subsel].reshape(dat.shape[0], -1)
        slices.append(subsel_dat)
    return np.hstack(slices)

sub_selections = {
    'hl_tracks': [
        ('jets', range(0, 2)),
        ('subjet1', list(range(0,15)) + list(range(16, 34)) + list(range(36, 40)) ),
        ('subjet2', list(range(0,15)) + list(range(16, 34)) + list(range(36, 40))),
    ],
    'mv2c10': [
        ('subjet1', [15]),
        ('subjet2', [15])
    ],
    'tracks': [
        ('jets', range(0,2)),
        ('tracks', list(range(0, 9)) + list(range(13, 29)))
    ]
}


def get_features(input_file, feature_name):
    ds_names, selection_lists = zip(*sub_selections[feature_name])
    data_list = [input_file[nm] for nm in ds_names]
    batch_size = data_list[0].shape[0]
    slices = []
    for dat, subsel in zip(data_list, selection_lists):
        # take a slice of the data, and then reshape it to flatten
        # everything past the first dimenstion
        subsel_dat = dat[:,subsel].reshape(dat.shape[0], -1)
        slices.append(subsel_dat)
    return np.hstack(slices)

def batch_generator(input_file, feature_group, batch_size=1000):
    total_size = input_file[feature_names[0]].shape[0]
    for start in range(0, total_size, batch_size):
        end = start + batch_size
        feature_arrays = {}
        for feature in feature_names:
            feature_arrays[feature] = flatten(input_file[feature])
        yield get_features(feature_arrays, feature_group)

def run():
    input_file_name = "small_test_raw_data_signal.h5"
    test_file = "test_data_dan.h5"

    save_file = File(test_file, 'w')

    with File(input_file_name) as input_file:
        for feature_group in sub_selections:
            for batch in batch_generator(input_file, feature_group, 10000):
                if feature_group not in save_file:
                    ds = save_file.create_dataset(
                        feature_group, batch.shape,
                        maxshape=(None,) + batch.shape[1:],
                        chunks=batch.shape,
                        dtype=np.float32)
                else:
                    ds = save_file[feature_group]
                old_size = ds.shape[0]
                new_size = old_size + batch.shape[0]
                ds.resize((new_size,) + ds.shape[1:])
                ds[old_size:new_size,...] = batch

if __name__ == '__main__':
    run()
