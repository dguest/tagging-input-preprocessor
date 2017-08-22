#!/usr/bin/env python3

"""
Tests for networks
"""
_help_arch_file = "NN archetecture file from Keras"
_help_vars_file = "Variable description file"
_help_hdf5_file = "NN weights file from Keras"

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from numpy import linspace
import numpy as np
import json
from math import isnan

def _get_args():
    parser = ArgumentParser(
        description=__doc__,
        formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument('archetecture_file', help=_help_arch_file)
    parser.add_argument('variables_file', help=_help_vars_file)
    parser.add_argument('hdf5_file', help=_help_hdf5_file)
    parser.add_argument('data_file', help='run on this file')
    return parser.parse_args()


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

    for pat in generate_test_pattern(args.data_file, input_dict=inputs):
        outputs = list(model.predict(pat))[0]
        print(outputs)

def generate_test_pattern(input_file, input_dict):

    pos_dict = {}
    scale = np.zeros((n_inputs,))
    offset = np.zeros((n_inputs,))
    for nnn, entry in enumerate(input_dict['inputs']):
        pos_dict[entry['name']] = nnn
        scale[nnn] = entry['scale']
        offset[nnn] = entry['offset']

    
    for key, value in zip(field_keys, field_values):
        input_pos = pos_dict[key]
        if isnan(value):
            value = input_dict['inputs'][input_pos]['default']
        raw_values[input_pos] = value

    normed_values = (raw_values + offset) * scale
    

if __name__ == '__main__':
    run()
