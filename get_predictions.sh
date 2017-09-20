#!/usr/bin/env bash

cd input_data
python3 flatten_hdf5.py raw_data.h5
python3 split_into_categories.py
rm temporary_flattened_data.h5
cd ..

./evaluate_network.py nn_data/architecture.json\
                  nn_data/hl_tracks_variable_description.json\
                  nn_data/weights.h5\
                  input_data/test_input.h5
