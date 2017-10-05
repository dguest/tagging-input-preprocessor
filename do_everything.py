#!/usr/bin/env python3

from h5py import File
import numpy as np

# _______________________________________________________________________
# part 1

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


def get_flat_array(input_file, output_file_name):
    with File(input_file) as test_h5:
        with File(output_file_name, 'w') as out_file:
            for feature_name in feature_names:
                print(feature_name)
                data = test_h5[feature_name]
                flat_ds = flatten(data)
                out_file.create_dataset(
                    feature_name,
                    flat_ds.shape,
                    data=flat_ds,
                    dtype=np.float32)

# ________________________________________________________________________
# Part 2

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
    ]
}


def get_features(input_file, output_file, feature_name):
    ds_names, selection_lists = zip(*sub_selections[feature_name])
    data_list = [input_file[nm] for nm in ds_names]
    N = data_list[0].shape[0]

    with File(output_file, 'w') as save_file:
        n_features = sum([len(l) for l in selection_lists])
        save_data = get_subselection(data_list, selection_lists)
        save_file.create_dataset(feature_name, (N, n_features),
                                 data=save_data, dtype=float)



def run():
    input_file = "small_test_raw_data_signal.h5"
    flat_file_name = "temporary_flattened_data_dan.h5"
    test_file = "test_data_dan.h5"
    get_flat_array(input_file, flat_file_name)
    with File(flat_file_name) as flat_file:
        get_features(flat_file, test_file, feature_name='hl_tracks')

if __name__ == '__main__':
    run()
