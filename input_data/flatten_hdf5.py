from __future__ import print_function

import numpy as np
import h5py
import utils

def create_dataset(open_file, feature_name, shape):
    try:
        return open_file.create_dataset(feature_name, shape, dtype=np.float32)
    except(RuntimeError):
        return open_file.get(feature_name)

# Merges many hdf5 files into one
new_file_dataset_name = "flattened_data.h5"
path = "./"
# This list can contain the names of many h5 files and it will merge them into one.
file_list = ['small_test_raw_data_signal.h5',]

f_names = []
for name in file_list:
    f_names.append(path + name)

feature_names = [u'clusters', u'jets', u'subjet1', u'subjet2', u'tracks']

files = []
for f_name in f_names:
    files.append(h5py.File(f_name, 'r'))

# Calculate total samples
round_down = 1.0  # This is in case you want to use only a percentage of the samples in each file, default is to use all (1.0) 
total_samples = utils.count_num_samples_from_hdf5_file_list(f_names, round_down)
print("total samples", total_samples)

new_hdf5 = h5py.File(path + new_file_dataset_name, 'w')
for feature_name in feature_names:
    print(feature_name)
    start = 0
    end = 0
    data = None
    old_names = None
    for f_name in f_names:
        print("loading %s" % f_name)
        f = h5py.File(f_name)
        data = f.get(feature_name)
        col_names = data.dtype.names
        assert data is not None
        N = int(total_samples)
        new_sizes = {u'clusters': (N, 5, 60),
                 u'jets': (N, 11),
                 u'subjet1': (N, 40),
                 u'subjet2': (N, 40),
                 u'subjet3': (N, 40),
                 u'tracks': (N, 29, 60),
                }

        # Create the new empty dataset
        if start == 0:
            create_dataset(new_hdf5, feature_name, new_sizes[feature_name])
         
        end = end + int(np.floor(data.shape[0]/round_down))

        save_data = new_hdf5.get(feature_name)
        # this could be made smaller to acomodate RAM requirements
        num_samples_this_file = int(np.floor(data.shape[0]/round_down))
        print(start, end, data.shape[0], end-start)
        if old_names is None:
            old_names = col_names
        assert old_names == col_names, old_names + col_names
        data = data[col_names] 
        if len(data.shape) == 1:
            data = data[:]
        elif len(data.shape) == 2:
            data = data[:, :]
        elif len(data.shape) == 3:
            data = data[:, :, :]   

        assert data is not None
        if len(data.shape) == 1:
            data = utils.flatten(data[0:num_samples_this_file])
            save_data[start:end] = data
        elif len(data.shape) == 2:
            data = utils.flatten(data[0:num_samples_this_file, :])
            save_data[start:end, :] = data
        elif len(data.shape) == 3:
            data = utils.flatten(data[0:num_samples_this_file, :, :])
            save_data[start:end, :, :] = data

        start = start + num_samples_this_file




