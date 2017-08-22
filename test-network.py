#!/usr/bin/env python3

"""
Tests for networks
"""
_help_arch_file = "NN archetecture file from Keras"
_help_vars_file = "Variable description file"
_help_hdf5_file = "NN weights file from Keras"

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from numpy import linspace
from numpy.lib.recfunctions import merge_arrays, rec_drop_fields
import numpy as np
import json
from math import isnan
from h5py import File

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
    def __init__(self, inputs):
        n_inputs = len(inputs)
        self.scale = np.zeros((n_inputs,))
        self.offset = np.zeros((n_inputs,))
        self.default = np.zeros((n_inputs,))
        self.pos_dict = {}
        self.input_list = [i['name'] for i in inputs]
        for nnn, entry in enumerate(inputs):
            self.pos_dict[entry['name']] = nnn
            self.scale[nnn] = entry['scale']
            self.offset[nnn] = entry['offset']
            self.default[nnn] = entry['default']

    def get_array(self, ds):
        drop = list(set(ds.dtype.names) - set(self.input_list))
        subarray = ds[self.input_list]
        ftype = [(n, float) for n in self.input_list]
        floated = subarray.astype(ftype).view(float).reshape(ds.shape + (-1,))
        normed_values = (floated + self.offset) * self.scale
        return normed_values


def flatten(ds):
    """
    """
    ftype = [(n, float) for n in ds.dtype.names]
    return ds.astype(ftype).view(float).reshape(ds.shape + (-1,))

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

    pos_dict = {}
    scale = np.zeros((n_inputs,))
    offset = np.zeros((n_inputs,))
    for nnn, entry in enumerate(inputs['inputs']):
        pos_dict[entry['name']] = nnn
        scale[nnn] = entry['scale']
        offset[nnn] = entry['offset']

    preprocessor = Preprocessor(inputs['inputs'])

    for pat in generate_test_pattern(args.data_file, input_dict=inputs):

        print(preprocessor.get_array(pat))
        return
        # normed_values = (raw_values + offset) * scale

        # outputs = list(model.predict(pat))[0]
        # print(outputs)

def generate_test_pattern(input_file, input_dict, chunk_size=2):

    with File(input_file, 'r') as h5file:
        for start in range(0, h5file['jets'].shape[0], chunk_size):
            sl = slice(start, start + chunk_size)

            subjets = []
            for sjn in [1,2]:
                subjet = np.asarray(h5file[f'subjet{sjn}'][sl])
                newnames = [f'subjet_{sjn}_{nm}' for nm in subjet.dtype.names]
                subjet.dtype.names = newnames
                subjets.append(subjet)
            fatjet = h5file['jets'][sl]
            newnames = [f'jet_{nm}' for nm in fatjet.dtype.names]
            fatjet.dtype.names = newnames

            yield merge_arrays((fatjet,*subjets), flatten=True)



if __name__ == '__main__':
    run()
