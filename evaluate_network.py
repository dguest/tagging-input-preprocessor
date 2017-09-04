#!/usr/bin/env python3

"""
Tests for networks
"""
_help_arch_file = "NN architecture file from Keras"
_help_vars_file = "Variable description file"
_help_hdf5_file = "NN weights file from Keras"

# NOTE: the code execution starts in `run()` below, skip down there!

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "%i" % 3

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from numpy import linspace
from numpy.lib.recfunctions import merge_arrays
import numpy as np
import json
from math import isnan
from h5py import File
from collections import Counter
import h5py

def _get_args():
    parser = ArgumentParser(
        description=__doc__,
        formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument('architecture_file', help=_help_arch_file)
    parser.add_argument('variables_file', help=_help_vars_file)
    parser.add_argument('hdf5_file', help=_help_hdf5_file)
    parser.add_argument('data_file', help='run on this file')
    return parser.parse_args()


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


    def get_array(self, data):
        flat_data = self.convert_2D_ndarray_to_numpy(data)
        flat_data = self.scale_and_center(flat_data)
        flat_data = self.replace_nans(flat_data)
        return flat_data

    def preprocess_data(self, data):
        flat_data = self.scale_and_center(data)
        flat_data = self.replace_nans(flat_data)
        return flat_data

    def replace_nans(self, data):
        nan_positions = np.isnan(data) | np.isinf(data)
        default_values = np.repeat(self.default[None,...], data.shape[0], axis=0)
        data[nan_positions] = default_values[nan_positions]

    def scale_and_center(self, data):
        return (data + self.offset) * self.scale

    def convert_2D_ndarray_to_numpy(self, data):
        """Returns a flat numpy array given a structured array"""
        # Get the variables we're actually going to use
        sub = data[self.input_list]
 
        # we need to convert everything into a float
        # first create the new datatype
        ftype = [(n, float) for n in self.input_list]

        # this magic replaces an array with named fields with an array
        # that has one extra dimension
        floated = sub.astype(ftype, casting='safe').view((float, len(ftype)))
        return floated

def load_keras_model(arch_file, weights_file):
    """ loads and initializes a keras model"""
    from keras.models import model_from_json, load_model
    model = load_model('nn_data/hl_tracks_model.keras_model')
    #with open(arch_file) as arch:
    #    model = model_from_json(arch)
    #    #model = model_from_json(''.join(arch.readlines()))
    #model.load_weights(weights_file)
    return model

def load_variable_names(variables_file_name):
    with open(variables_file_name) as variables_file:
        inputs = json.loads(''.join(variables_file.readlines()))
    return inputs

def run_julian():
    args = _get_args()
    model = load_keras_model(args.architecture_file, args.hdf5_file)
    variable_names = load_variable_names(args.variables_file)
    num_inputs = model.layers[0].input_shape[1]
    assert num_inputs == len(variable_names['inputs'])
    preprocessor = Preprocessor(variable_names['inputs'])

    data = load_julian_processed_hdf5_data(file_name= args.data_file, feature='hl_tracks')
    array = preprocessor.preprocess_data(data)
    outputs = model.predict(array)[:,0]  # TODO should it be [:, 1] instead?
    print(outputs)

"""
def run():
#main function call for this script

    # read in the command line options
    args = _get_args()

    # load the keras model
    model = load_keras_model(args.architecture_file, args.hdf5_file)

    # load in the names of the variables that we're feeding the network
    variable_names = load_variable_names(args.variables_file)

    # quick sanity check to make sure the model matches the variable names file
    num_inputs = model.layers[0].input_shape[1]
    assert num_inputs == len(variable_names['inputs'])

    # the preprocessor handles all the nan replacement, scaling, shifts, etc
    preprocessor = Preprocessor(variable_names['inputs'])

    # This is the main loop over jets. We keep track of how often the
    # values we compute from Keras disagree with the ones stored in
    # the input file.
    n_diff = Counter()
    for pattern in generate_test_pattern(args.data_file, input_dict=variable_names):

        # get inputs for keras and feed them to the model
        array = preprocessor.get_array(pattern)
        outputs = model.predict(array)[:,0]

        # look up the pre-computed values
        discrim = pattern['jet_discriminant']

        # print some information if there's any disagreement. Also
        # keep track of how often it happens
        not_close = ~np.isclose(outputs, discrim)
        if np.any(not_close):
            keras = outputs[not_close]
            lwtnn = discrim[not_close]
            diff = lwtnn - keras
            for ker, lw, dif in zip(keras, lwtnn, diff):
                print(f'keras: {ker:.3}')
                print(f'keras: {ker:.3}, lwtnn: {lw:.3}, diff: {dif:.3}')
            n_diff['diffs'] += diff.size
        n_diff['total'] += pattern.size

    # Print summary information
    print('total with difference: {} of {}, ({:.3%})'.format(
        n_diff['diffs'], n_diff['total'], n_diff['diffs']/n_diff['total']))
"""

def load_julian_processed_hdf5_data(feature, file_name="./input_data/small_test_categorized_data_signal.h5"):
    with h5py.File(file_name, 'r') as hf:
        data = hf.get("/%s/%s" % (feature, 'test'))
    return data

def load_raw_hdf5_data(feature, file_name="./input_data/small_test_data_signal.h5", num_samples=None):
    with h5py.File(file_name, 'r') as hf:
        data = hf.get(feature)
    return data
"""
def generate_test_pattern(input_file, input_dict, chunk_size=200):

    #This spits out structured arrays, it may have to be changed to flatten
    #the 2d arrays of tracks and clusters

    with File(input_file, 'r') as h5file:
        for start in range(0, h5file['jets'].shape[0], chunk_size):
            sl = slice(start, start + chunk_size)

            # build a list of subjets
            subjets = []
            for sjn in [1,2]:

                subjet = np.asarray(h5file[f'subjet{sjn}'][sl])

                # we rename the variables in the subjet (since we're
                # flattening them and the names will clash with the
                # other jets)
                newnames = [f'subjet_{sjn}_{nm}' for nm in subjet.dtype.names]
                subjet.dtype.names = newnames

                subjets.append(subjet)

            # grab the fatjet and rename the fields
            fatjet = h5file['jets'][sl]
            newnames = [f'jet_{nm}' for nm in fatjet.dtype.names]
            fatjet.dtype.names = newnames

            yield merge_arrays((fatjet,*subjets), flatten=True)
"""


if __name__ == '__main__':
    run_julian()
    #run()
