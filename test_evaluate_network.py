#!/usr/bin/env python3

import numpy as np
from evaluate_network import _get_args, load_keras_model, load_variable_names, Preprocessor, load_julian_processed_hdf5_data
from evaluate_network import load_mean_and_std_vectors

_help_arch_file = "NN architecture file from Keras"
_help_vars_file = "Variable description file"
_help_hdf5_file = "NN weights file from Keras"

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


def test_2D_ndarray_to_numpy():
    pass

def test_3D_ndarray_to_numpy():
    pass

def test_offset():
    pass

def test_scaling():
    pass

def test_missing_value_replacement():
    pass

def test_loading_and_applying_scaling_from_Dan_json_vs_Julian_numpy():
    def predictions_using_julian_mean_and_std():
        args = _get_args()
        model = load_keras_model(args.architecture_file, args.hdf5_file)
        variable_names = load_variable_names(args.variables_file)
        num_inputs = model.layers[0].input_shape[1]
        assert num_inputs == len(variable_names['inputs'])
        preprocessor = Preprocessor(variable_names['inputs'])
        data = load_julian_processed_hdf5_data(file_name= args.data_file, feature='hl_tracks')
        mean_vector, std_vector = load_mean_and_std_vectors(feature='hl_tracks')
        assert mean_vector is not None
        assert std_vector is not None
        std_vector[std_vector==0] = 1
        array = preprocessor.preprocess_data(data, mean_vector, std_vector)
        assert array is not None
        outputs = model.predict(array) 
        assert outputs is not None
        print(outputs)
        outputs = outputs[:,0]
        labels = np.round(outputs)
        return labels
    def predictions_using_dan_offset_and_scale():
        args = _get_args()
        model = load_keras_model(args.architecture_file, args.hdf5_file)
        variable_names = load_variable_names(args.variables_file)
        num_inputs = model.layers[0].input_shape[1]
        assert num_inputs == len(variable_names['inputs'])
        preprocessor = Preprocessor(variable_names['inputs'])
        data = load_julian_processed_hdf5_data(file_name= args.data_file, feature='hl_tracks')
        mean_vector, std_vector = None, None
        array = preprocessor.preprocess_data(data, mean_vector, std_vector)
        assert array is not None
        outputs = model.predict(array) 
        assert outputs is not None
        outputs = outputs[:,0]
        labels = np.round(outputs)
        print(outputs)
        return labels
    #print(np.isclose(predictions_using_julian_mean_and_std(), predictions_using_dan_offset_and_scale()))
    assert np.all(np.isclose(predictions_using_julian_mean_and_std(), predictions_using_dan_offset_and_scale()))

def test_predictions_from_raw_julian_h5_vs_preprocesses_julian_h5():
    pass

 __name__ == "__main__":
    test_loading_and_applying_scaling_from_Dan_json_vs_Julian_numpy()



