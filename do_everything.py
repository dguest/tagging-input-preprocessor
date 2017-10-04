#!/usr/bin/env python3

from h5py import File
from input_data import utils
import numpy as np

# This is in case you want to use only a percentage of the samples in each file, default is to use all (1.0)
round_down = 1.0

feature_names = [u'clusters', u'jets', u'subjet1', u'subjet2', u'tracks']

def get_flat_array(input_file, output_file_name):
    total_samples = utils.count_num_samples_from_hdf5_file_list(
        [input_file], round_down)
    with File(input_file) as test_h5:
        for feature_name in feature_names:
            print(feature_name)
            start = 0
            end = 0
            data = test_h5[feature_name]
            col_names = data.dtype.names
            end = end + int(np.floor(data.shape[0]/round_down))
            num_samples_this_file = int(np.floor(data.shape[0]/round_down))
            # data = data[col_names][:]
            # if len(data.shape) == 1:
            #     data = data[:]
            # elif len(data.shape) == 2:
            #     data = data[:, :]
            # elif len(data.shape) == 3:
            #     data = data[:, :, :]

            N = int(total_samples)
            new_sizes = {
                u'clusters': (N, 5, 60),
                u'jets': (N, 11),
                u'subjet1': (N, 40),
                u'subjet2': (N, 40),
                u'subjet3': (N, 40),
                u'tracks': (N, 29, 60),
            }

            with File(output_file_name, 'w') as out_file:
                save_data = out_file.create_dataset(
                    feature_name,
                    new_sizes[feature_name],
                    dtype=np.float32)

                if len(data.shape) == 1:
                    data = utils.flatten(data[0:num_samples_this_file])
                    save_data[start:end] = data
                elif len(data.shape) == 2:
                    data = utils.flatten(data[0:num_samples_this_file, :])
                    save_data[start:end, :] = data
                elif len(data.shape) == 3:
                    data = utils.flatten(data[0:num_samples_this_file, :, :])
                    save_data[start:end, :, :] = data


def run():
    input_file = "small_test_raw_data_signal.h5"
    output_file = "temporary_flattened_data_dan.h5"
    get_flat_array(input_file, output_file)

if __name__ == '__main__':
    run()
