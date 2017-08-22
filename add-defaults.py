#!/usr/bin/env python3

"""add default values for input variables"""

import argparse
import json
import sys

def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_file')
    return parser.parse_args()

def run():
    args = get_args()
    with open(args.input_file, 'r') as in_file:
        vars_list = json.load(in_file)

    for variable in vars_list['inputs']:
        variable['default'] = -variable['offset']

    sys.stdout.write(json.dumps(vars_list, indent=2))


if __name__ == '__main__':
    run()
