#!/usr/bin/env python
from __future__ import print_function

import numpy as np
from evaluate_network import _get_args, load_keras_model, load_variable_information, Preprocessor, load_julian_processed_hdf5_data
from evaluate_network import load_mean_and_std_vectors, extract_values_as_list, remove_initial_string_from_elements, replace_values_in_dicts
from evaluate_network import load_raw_hdf5_data

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
        variable_names = load_variable_information(args.variables_file)
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
        variable_names = load_variable_information(args.variables_file)
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

def test_predictions_from_flattened_julian_h5_vs_preprocesses_julian_h5():
    def predictions_using_dan_offset_and_scale():
        args = _get_args()
        model = load_keras_model(args.architecture_file, args.hdf5_file)
        variable_information = load_variable_information(args.variables_file)
        num_inputs = model.layers[0].input_shape[1]
        assert num_inputs == len(variable_information['inputs'])
        variable_names = extract_values_as_list(variable_information['inputs'], 'name')
        preprocessor = Preprocessor(variable_information['inputs'])
        data = load_julian_processed_hdf5_data(file_name= args.data_file, feature='hl_tracks')
        mean_vector, std_vector = None, None
        print('raw data')
        print(variable_names[0:5])
        print(data[0, 0:5])
        array = preprocessor.preprocess_data(data, mean_vector, std_vector)
        assert array is not None
        print('preprocessed data')
        print(array[0, 0:5])
        outputs = model.predict(array) 
        assert outputs is not None
        print(outputs)
        outputs = outputs[:,0]
        labels = np.round(outputs)
        print(labels)
        return labels   
    def predictions_from_flattened_julian_h5():
        args = _get_args()
        model = load_keras_model(args.architecture_file, args.hdf5_file)
        variable_information = load_variable_information(args.variables_file)
        num_inputs = model.layers[0].input_shape[1]
        assert num_inputs == len(variable_information['inputs'])

        jet = load_raw_hdf5_data('jets', file_name='./input_data/small_test_flattened_data_signal.h5')
        sub_jet_1 = load_raw_hdf5_data('subjet1', file_name='./input_data/small_test_flattened_data_signal.h5')
        sub_jet_2 = load_raw_hdf5_data('subjet2', file_name='./input_data/small_test_flattened_data_signal.h5')

        print(jet.shape, sub_jet_1.shape, sub_jet_1.shape)

        sub_selection_0 = range(1,3)
        sub_selection_1 = list(range(0,15)) + list(range(16, 34)) + list(range(36, 40)) 
        sub_selection_2 = list(range(0,15)) + list(range(16, 34)) + list(range(36, 40)) 

        jet = jet[:, sub_selection_0]
        sub_jet_1 = sub_jet_1[:, sub_selection_1]
        sub_jet_2 = sub_jet_2[:, sub_selection_2]

        print("after subselection")
        print(jet.shape, sub_jet_1.shape, sub_jet_1.shape)

        print("raw flat jet")
        print(jet.shape)
        print(jet[0])
        #print("this means there is a bug in preprocessor.convert_2D_ndarray_to_numpy")

        data = np.hstack((jet, sub_jet_1))
        data = np.hstack((data, sub_jet_2))
        data = data[0:8, :]
        print("raw flat stacked inputs")
        print(data[0, 0:5])
        print(data.shape)

        mean_vector, std_vector = None, None
        #variable_information = kinematic_information+sub_jet_1_information+sub_jet_2_information
        #assert len(variable_information) == data.shape[1], len(variable_information)
        preprocessor = Preprocessor(variable_information['inputs'])
        array = preprocessor.preprocess_data(data, mean_vector, std_vector)
        assert array is not None
        print("scaled inputs")
        print(array[0, 0:5])
        outputs = model.predict(array)
        assert outputs is not None
        print("outputs")
        print(outputs)
        outputs = outputs[:,0]
        labels = np.round(outputs)
        print(labels)
        return labels
    print("my preprocessing")
    predictions_using_dan_offset_and_scale()
    print("Flattened data with Dan's preprocessing")
    predictions_from_flattened_julian_h5()

def test_predictions_from_raw_julian_h5_vs_preprocesses_julian_h5():
    def predictions_using_dan_offset_and_scale():
        args = _get_args()
        model = load_keras_model(args.architecture_file, args.hdf5_file)
        variable_information = load_variable_information(args.variables_file)
        num_inputs = model.layers[0].input_shape[1]
        assert num_inputs == len(variable_information['inputs'])
        variable_names = extract_values_as_list(variable_information['inputs'], 'name')
        preprocessor = Preprocessor(variable_information['inputs'])
        data = load_julian_processed_hdf5_data(file_name= args.data_file, feature='hl_tracks')
        mean_vector, std_vector = None, None
        print('raw data')
        print(variable_names[0:5])
        print(data[0, 0:5])
        array = preprocessor.preprocess_data(data, mean_vector, std_vector)
        assert array is not None
        print('preprocessed data')
        print(array[0, 0:5])
        outputs = model.predict(array) 
        assert outputs is not None
        print(outputs)   
        outputs = outputs[:,0]
        labels = np.round(outputs)
        print(labels)
        return labels
    def predictions_from_raw_julian_h5():
        args = _get_args()
        model = load_keras_model(args.architecture_file, args.hdf5_file)
        variable_information = load_variable_information(args.variables_file)
        num_inputs = model.layers[0].input_shape[1]
        assert num_inputs == len(variable_information['inputs'])

        kinematic_information = variable_information['inputs'][0:2]
        sub_jet_1_information = variable_information['inputs'][2:39]
        sub_jet_2_information = variable_information['inputs'][39:76]

        kinematic_names = extract_values_as_list(kinematic_information, 'name')
        sub_jet_1_names = extract_values_as_list(sub_jet_1_information, 'name')
        sub_jet_2_names = extract_values_as_list(sub_jet_2_information, 'name')

        kinematic_names = remove_initial_string_from_elements(kinematic_names, initial_string="jet_")
        sub_jet_1_names = remove_initial_string_from_elements(sub_jet_1_names, initial_string="subjet_1_")
        sub_jet_2_names = remove_initial_string_from_elements(sub_jet_2_names, initial_string="subjet_2_")

        kinematic_information = replace_values_in_dicts(kinematic_information, kinematic_names, 'name')
        sub_jet_1_information = replace_values_in_dicts(sub_jet_1_information, sub_jet_1_names, 'name')
        sub_jet_1_information = replace_values_in_dicts(sub_jet_2_information, sub_jet_2_names, 'name')

        kin_preprocessor = Preprocessor(kinematic_information)
        sub_jet_1_preprocessor = Preprocessor(sub_jet_1_information)
        sub_jet_2_preprocessor = Preprocessor(sub_jet_2_information)

        jet = load_raw_hdf5_data('jets')
        sub_jet_1 = load_raw_hdf5_data('subjet1')
        sub_jet_2 = load_raw_hdf5_data('subjet2')

        print("###################### here you can see you the behaviour is different #########################")
        print("raw jet and names")
        print('jet.dtype.names but this ordering is wrong if you just call it like that')
        print('jet.dtype.names')
        print(jet[0])
        
        # I think here is the bug
        print("name, jet[0][name], jet[name][0]")
        for name in jet.dtype.names:
            print(name, jet[0][name], jet[name][0])
        print("the correct way is jet[name][0] since otherwise the weight would be negative")
        print("################################################################################################")

        jet = kin_preprocessor.convert_2D_ndarray_to_numpy(jet)
        sub_jet_1 = sub_jet_1_preprocessor.convert_2D_ndarray_to_numpy(sub_jet_1)
        sub_jet_2 = sub_jet_2_preprocessor.convert_2D_ndarray_to_numpy(sub_jet_2)

        print("raw flat jet")
        print(jet.shape)
        print(jet[0])
        print("this means there is a bug in preprocessor.convert_2D_ndarray_to_numpy")

        data = np.hstack((jet, sub_jet_1))
        data = np.hstack((data, sub_jet_2))
        data = data[0:8, :]
        print("raw flat stacked inputs")
        print(data[0, 0:5])
        print(data.shape)

        mean_vector, std_vector = None, None
        variable_information = kinematic_information+sub_jet_1_information+sub_jet_2_information
        assert len(variable_information) == data.shape[1], len(variable_information)
        preprocessor = Preprocessor(variable_information)
        array = preprocessor.preprocess_data(data, mean_vector, std_vector)
        assert array is not None
        print("scaled inputs")
        print(array[0, 0:5])
        outputs = model.predict(array)
        assert outputs is not None
        print("outputs")
        print(outputs)
        outputs = outputs[:,0]
        labels = np.round(outputs)
        print(labels)
        return labels
    print("my preprocessing")
    predictions_using_dan_offset_and_scale()
    print("Dan's preprocessing")
    predictions_from_raw_julian_h5()

if __name__ == "__main__":
    #test_predictions_from_flattened_julian_h5_vs_preprocesses_julian_h5()
    test_predictions_from_raw_julian_h5_vs_preprocesses_julian_h5()


