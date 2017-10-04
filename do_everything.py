#!/usr/bin/env python3

from h5py import File
from input_data import utils
import numpy as np

# _______________________________________________________________________
# part 1

# feature_names = [u'clusters', u'jets', u'subjet1', u'subjet2', u'tracks']
feature_names = [u'jets']

def get_flat_array(input_file, output_file_name):
    with File(input_file) as test_h5:
        with File(output_file_name, 'w') as out_file:
            for feature_name in feature_names:
                print(feature_name)
                data = test_h5[feature_name]
                fields = data.dtype.names + (Ellipsis,)
                print(data[fields][0])
                print(data[data.dtype.names][0])
                print(data[:][0])
                print(np.asarray(data)[0])
                flat_ds = utils.flatten(np.asarray(data[data.dtype.names]))
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
    if len(data.shape) == 1:
        return data[start:end]
    elif len(data.shape) == 2:
        return data[start:end, sub_selection]
    elif len(data.shape) == 3:
        return data[start:end, sub_selection, :]

def copy_in_batches(data, save_data, sub_selection):
    assert data is not None
    assert save_data is not None

    rounded_num_samples = save_data.shape[0]
    batch_size = 100

    if data is list:
        assert len(data) == len(sub_selection), "a sub_selection must be specified for each dataset"

    for start, end in zip(range(0, rounded_num_samples, batch_size),
                          range(batch_size, rounded_num_samples+batch_size, batch_size)):
        if data.__class__ is list:
            assert len(data) == len(sub_selection), "a sub_selection must be specified for each dataset"
            if len(data) == 2:
                data_1 = data[0]
                data_2 = data[1]
                sub_selection_1 = sub_selection[0]
                sub_selection_2 = sub_selection[1]

                temp_data_1 = sub_select_data(data_1, start, end, sub_selection_1)
                temp_data_2 = sub_select_data(data_2, start, end, sub_selection_2)

                temp_data_1 = flatten_3d_into_2d(temp_data_1)
                temp_data_2 = flatten_3d_into_2d(temp_data_2)

                mini_batch = np.hstack((temp_data_1, temp_data_2))

            elif len(data) == 3:
                data_1 = data[0]
                data_2 = data[1]
                data_3 = data[2]
                sub_selection_1 = sub_selection[0]
                sub_selection_2 = sub_selection[1]
                sub_selection_3 = sub_selection[2]
                temp_data_1 = sub_select_data(data_1, start, end, sub_selection_1)
                temp_data_2 = sub_select_data(data_2, start, end, sub_selection_2)
                temp_data_3 = sub_select_data(data_3, start, end, sub_selection_3)
                temp_data_1 = flatten_3d_into_2d(temp_data_1)
                temp_data_2 = flatten_3d_into_2d(temp_data_2)
                temp_data_3 = flatten_3d_into_2d(temp_data_3)
                merged_1_2 = np.hstack((temp_data_1, temp_data_2))
                mini_batch = np.hstack((merged_1_2, temp_data_3))

        else:
            mini_batch = sub_select_data(data, start, end, sub_selection)
            mini_batch = flatten_3d_into_2d(mini_batch)

        num_dims = len(mini_batch.shape)

        if num_dims == 1:
            save_data[start:end] = mini_batch
        elif num_dims == 2:
            save_data[start:end, :] = mini_batch
        elif num_dims == 3:
            save_data[start:end, :, :] = mini_batch


def get_high_level_tracks(input_file, output_file):
    sub_selection_0 = range(0, 2)
    sub_selection_1 = list(range(0,15)) + list(range(16, 34)) + list(range(36, 40))
    sub_selection_2 = list(range(0,15)) + list(range(16, 34)) + list(range(36, 40))
    with File(input_file) as open_file:
        data_0 = open_file['jets/']
        data_1 = open_file['subjet1/']
        data_2 = open_file['subjet2/']

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
