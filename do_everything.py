#!/usr/bin/env python3

from h5py import File
from input_data import utils
import numpy as np

feature_names = [u'clusters', u'jets', u'subjet1', u'subjet2', u'tracks']

def get_flat_array(input_file, output_file_name):
    with File(input_file) as test_h5:
        for feature_name in feature_names:
            print(feature_name)
            data = test_h5[feature_name]

            flat_ds = utils.flatten(np.asarray(data))
            with File(output_file_name, 'w') as out_file:
                save_data = out_file.create_dataset(
                    feature_name,
                    flat_ds.shape,
                    data=flat_ds,
                    dtype=np.float32)

def run():
    input_file = "small_test_raw_data_signal.h5"
    output_file = "temporary_flattened_data_dan.h5"
    get_flat_array(input_file, output_file)

if __name__ == '__main__':
    run()
