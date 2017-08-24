#!/usr/bin/env python3

"""
Tests for networks
"""
_help_arch_file = "NN archetecture file from Keras"
_help_vars_file = "Variable description file"
_help_hdf5_file = "NN weights file from Keras"

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from numpy import linspace
from numpy.lib.recfunctions import merge_arrays
import numpy as np
import json
from math import isnan
from h5py import File
from collections import Counter

def _get_args():
    parser = ArgumentParser(
        description=__doc__,
        formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument('archetecture_file', help=_help_arch_file)
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
        n_inputs = len(inputs)
        self.scale = np.zeros((n_inputs,))
        self.offset = np.zeros((n_inputs,))
        self.default = np.zeros((n_inputs,))
        self.input_list = [i['name'] for i in inputs]
        for nnn, entry in enumerate(inputs):
            self.scale[nnn] = entry['scale']
            self.offset[nnn] = entry['offset']
            self.default[nnn] = entry['default']


    def get_array(self, ds):
        """Returns a flat numpy array given a structured array"""

        # Get the variables we're actually going to use
        sub = ds[self.input_list]

        # we need to convert everything into a float
        # first create the new datatype
        ftype = [(n, float) for n in self.input_list]
        # this magic replaces an array with named fields with an array
        # that has one extra dimension
        floated = sub.astype(ftype, casting='safe').view((float, len(ftype)))

        # replace the inf and nan fields with defaults
        nans = np.isnan(floated) | np.isinf(floated)
        # NOTE: this part may need some rewriting to work with 2d arrays!
        defaults = np.repeat(self.default[None,...], ds.shape[0], axis=0)
        floated[nans] = defaults[nans]

        # shift and normalize data
        normed_values = (floated + self.offset) * self.scale

        return normed_values



def run():
    args = _get_args()

    # keras loads slow, do the loading here
    from keras.models import model_from_json

    with open(args.archetecture_file) as arch:
        model = model_from_json(''.join(arch.readlines()))
    model.load_weights(args.hdf5_file)

    with open(args.variables_file) as variables_file:
        inputs = json.loads(''.join(variables_file.readlines()))

    n_inputs = model.layers[0].input_shape[1]
    assert n_inputs == len(inputs['inputs'])

    preprocessor = Preprocessor(inputs['inputs'])

    n_diff = Counter()
    for pattern in generate_test_pattern(args.data_file, input_dict=inputs):

        array = preprocessor.get_array(pattern)
        outputs = model.predict(array)[:,0]
        discrim = pattern['jet_discriminant']
        not_close = ~np.isclose(outputs, discrim)
        if np.any(not_close):
            keras = outputs[not_close]
            lwtnn = discrim[not_close]
            diff = lwtnn - keras
            for ker, lw, dif in zip(keras, lwtnn, diff):
                print(f'keras: {ker:.3}, lwtnn: {lw:.3}, diff: {dif:.3}')
            n_diff['diffs'] += diff.size
        n_diff['total'] += pattern.size

    print('total with difference: {} of {}, ({:.3%})'.format(
        n_diff['diffs'], n_diff['total'], n_diff['diffs']/n_diff['total']))



def generate_test_pattern(input_file, input_dict, chunk_size=200):
    """
    This spits out structured arrays, it may have to be changed to flatten
    the 2d arrays of tracks and clusters
    """
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



if __name__ == '__main__':
    run()
