from __future__ import print_function

import numpy as np
import h5py

raw = h5py.File('small_test_raw_data_signal.h5', 'r')
flat = h5py.File('temporary_flattened_data.h5', 'r')

for key in raw.keys():
    if key != 'subjet3':
        print(key)
        raw_c = raw.get(key)
        flat_c = flat.get(key)
        # load values into memory
        raw_samples = raw_c[:]
        flat_samples = flat_c[:]
        same = True
        if len(raw_samples.shape) + 1 == len(flat_samples.shape):
            if len(raw_samples.shape) == 1:
                raw_samples = raw_samples[0]
                flat_samples = flat_samples[0, :]
            elif len(raw_samples.shape) == 2:
                raw_samples = raw_samples[0, 0]
                flat_samples = flat_samples[0, :, 0]

        for e1, e2 in zip(raw_samples, flat_samples):
            if not e1==e2 and not (np.isnan(e1) and np.isnan(e2)):
                print(e1, e2)
                same=False
        print(same)
    

