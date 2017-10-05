#!/usr/bin/env python3
from h5py import File
import numpy as np
from sys import argv

if len(argv) != 3:
    exit(f'usage {__file__} input_file output_file')

with File(argv[2], 'w') as out_file:
    out_file.create_dataset('np', data=np.load(argv[1]), dtype=np.float32)
