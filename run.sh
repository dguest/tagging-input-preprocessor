#!/usr/bin/env bash

./evaluate_network.py nn_data/architecture.json\
                  nn_data/hl_tracks_variable_description.json\
                  nn_data/weights.h5\
                  input_data/small_test_categorized_data_signal.h5
                  #input_data/test_input.h5
