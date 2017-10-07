#!/usr/bin/env bash

cd input_data
rm test_data.h5
python flatten_hdf5.py small_test_raw_data_signal.h5
python split_into_categories.py
rm temporary_flattened_data.h5
cd ..

./evaluate_network.py nn_data/architecture.json\
                  nn_data/hl_tracks_variable_description.json\
                  nn_data/weights.h5\
                  input_data/test_data.h5
