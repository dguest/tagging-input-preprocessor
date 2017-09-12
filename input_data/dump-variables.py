#!/usr/bin/env python3

import numpy as np
import h5py
input_file = 'small_test_raw_data_signal.h5'

jet_selection = range(1,3)
sub_selection_1 = list(range(0,15)) + list(range(16, 34)) + list(range(36, 40))
sub_selection_2 = list(range(0,15)) + list(range(16, 34)) + list(range(36, 40))

with h5py.File(input_file, 'r') as h5file:
    jet_names = h5file['jets'].dtype.names
    subjet_names = h5file['subjet1'].dtype.names
    print([jet_names[i] for i in jet_selection])
    print([subjet_names[i] for i in sub_selection_1])

