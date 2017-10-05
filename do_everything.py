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

def flatten_3d_into_2d(data):
    if len(data.shape) == 3:
        reshaped_data = data.reshape(data.shape[0], data.shape[1]*data.shape[2])
        return reshaped_data
    else:
        return data

def sub_select_data(data, start, end, sub_selection):
    return data[:,sub_selection,...]

def copy_in_batches(data, save_data, sub_selection):
    batch_size = save_data.shape[0]
    batches = []
    for dat, subsel in zip(data, sub_selection):
        subsel_dat = sub_select_data(dat, 0, 0, subsel)
        batches.append(flatten_3d_into_2d(subsel_dat))
    save_data[...] = np.hstack(batches)

def get_high_level_tracks(input_file, output_file):
    sub_selection_0 = range(0, 2)
    sub_selection_1 = list(range(0,15)) + list(range(16, 34)) + list(range(36, 40))
    sub_selection_2 = list(range(0,15)) + list(range(16, 34)) + list(range(36, 40))
    feature_name = 'hl_tracks'
    with File(input_file) as open_file:
        data_0 = open_file['jets/']
        data_1 = open_file['subjet1/']
        data_2 = open_file['subjet2/']
        N = data_1.shape[0]

        with File(output_file, 'w') as save_file:
            save_data = save_file.create_dataset(feature_name, (N, 2+37*2),
                                                 dtype=float)
            copy_in_batches([data_0, data_1, data_2], save_data, [sub_selection_0, sub_selection_1, sub_selection_2])


def run():
    input_file = "small_test_raw_data_signal.h5"
    flat_file = "temporary_flattened_data_dan.h5"
    test_file = "test_data_dan.h5"
    get_flat_array(input_file, flat_file)
    get_high_level_tracks(flat_file, test_file)

if __name__ == '__main__':
    run()
