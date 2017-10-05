#!/usr/bin/env python

"""
This should do everything to run Julian's network
"""

from h5py import File
import numpy as np
from argparse import ArgumentParser
import json

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
            feature_arrays[feature] = flatten(input_file[feature][start:end])
        yield get_features(feature_arrays, feature_group)

def multi_file_generator(file_list, feature_group, batch_size=10000):
    for file_name in file_list:
        with File(file_name) as h5_file:
            for batch in batch_generator(h5_file, feature_group, batch_size):
                yield batch

def get_args():
    input_files = ["small_test_raw_data_signal.h5"]
    vars_file = "../nn_data/hl_tracks_variable_description.json"
    arch = "../nn_data/architecture.json"
    weights = "../nn_data/weights.h5"
    feature_groups = sub_selections.keys()
    d = dict(help='default: %(default)s')
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('input_files', default=input_files, nargs='*')
    parser.add_argument('-v','--variables-file',
                        default=vars_file, **d)
    parser.add_argument('-a','--architecture-file', default=arch, **d)
    parser.add_argument('-w', '--weights-file', default=weights, **d)
    parser.add_argument('-f', '--feature-group', default='hl_tracks',
                        choices=list(feature_groups), **d)
    parser.add_argument('-b', '--batch-size', type=int, default=10000, **d)
    return parser.parse_args()

def run():
    args = get_args()

    from keras.models import model_from_json

    with open(args.architecture_file) as arch:
        model = model_from_json(arch.read())
    model.load_weights(args.weights_file)
    with open(args.variables_file) as v_file:
        inputs = json.loads(''.join(v_file.readlines()))
    preproc = Preprocessor(inputs['inputs'])

    test_file = "test_data_dan.h5"
    save_file = File(test_file, 'w')

    batch_size = args.batch_size
    output_ds = save_file.create_dataset(
        'output', (0,), maxshape=(None,), chunks=(batch_size,),
        dtype=np.float32)

    feature_group = args.feature_group
    input_files = args.input_files
    for batch in multi_file_generator(input_files, feature_group, batch_size):
        if feature_group not in save_file:
            ds = save_file.create_dataset(
                feature_group, (0,) + batch.shape[1:],
                maxshape=(None,) + batch.shape[1:],
                chunks=batch.shape,
                dtype=np.float32)
        else:
            ds = save_file[feature_group]
        old_size = ds.shape[0]
        new_size = old_size + batch.shape[0]
        ds.resize((new_size,) + ds.shape[1:])
        ds[old_size:new_size,...] = batch

        output = model.predict(preproc.preprocess_data(batch))
        output_ds.resize((new_size,))
        output_ds[old_size:new_size] = output[:,0]

    save_file.close()

class Preprocessor:
    """
    This class handles all the conversion between the original HDF5
    file and the numpy arrays we feed the network.
    """
    def __init__(self, inputs):
        num_inputs = len(inputs)
        self.scale = np.zeros((num_inputs,))
        self.offset = np.zeros((num_inputs,))
        self.default = np.zeros((num_inputs,))
        self.input_list = [i['name'] for i in inputs]
        for nnn, entry in enumerate(inputs):
            self.scale[nnn] = entry['scale']
            self.offset[nnn] = entry['offset']
            self.default[nnn] = entry['default']

    def preprocess_data(self, data):
        flat_data = self.scale_and_center(data)
        flat_data = self.replace_nans(flat_data)
        return flat_data

    def replace_nans(self, data):
        nan_positions = np.isnan(data) | np.isinf(data)
        default_values = np.repeat(
            self.default[None,...], data.shape[0], axis=0)
        data[nan_positions] = default_values[nan_positions]
        return data

    def scale_and_center(self, data):
        return (data + self.offset) * self.scale


if __name__ == '__main__':
    run()
