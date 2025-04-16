#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert the C++ results to HDF5 files.

Note that the field 'xmean' stores all runs (repeats are stored along axis 0).
In order to average over the 'num_repeats' runs, use
    xmean = np.mean(data['xmean'], axis=0)

List of all key-value pairs in a HDF5 file with
    data = h5py.File(filename)
    list(data.items())

@author: joachim
"""

import h5py
import numpy as np


def main():
    """Convert ratchet simulation results to HDF5"""
    print(__doc__)
    filename = "ratchet_dimensionless/results_96x100"
    with open(filename + ".txt", 'r') as file:
        lines = [file.readline().replace('\n', '').split(',')
                 for _ in range(5)]
        header, h_vals, pot_header, pot_vals, data_header = lines
        ratchet_params = {}
        for i in range(len(header)):
            key, val = header[i], h_vals[i]
            if '.' in val:   # value contains a dot --> float
                ratchet_params[key] = float(val)
            else:
                ratchet_params[key] = int(val)

        pot_params = {}
        for i in range(len(pot_header)):
            key, val = pot_header[i], pot_vals[i]
            pot_params[key] = float(val)

    shape = (ratchet_params['num_a'], ratchet_params['num_theta'])
    size = shape[0] * shape[1]
    data = {data_header[0] : np.zeros(shape),
            data_header[1] : np.zeros(shape),
            data_header[-1] : np.zeros((ratchet_params['num_repeat'], *shape))}

    # load 'a' and 'theta' data from file
    at_data = np.loadtxt(filename + ".txt",
                         skiprows=len(lines),
                         max_rows=size,
                         usecols=[0, 1],
                         delimiter=',')

    # get 'a' and 'theta' (using 'data_header' to avoid name conflicts)
    for i, key in enumerate(data_header[:-1]):
        data[key] = at_data[:size, i].reshape(shape)

    # get all the 'x' values
    x_data = np.loadtxt(filename + ".txt", skiprows=len(lines),
                        usecols=[-1], delimiter=',')
    data[data_header[-1]] = x_data.reshape(data[data_header[-1]].shape)

    with h5py.File(filename + ".hdf5", "w") as file:
        for key, val in data.items():
            file.create_dataset(key, data=val)

        for key, val in ratchet_params.items():
            file.create_dataset("ratchet_params/" + key, data=val)

        for key, val in pot_params.items():
            file.create_dataset("pot_params/" + key, data=val)

    return 0

if __name__ == "__main__":
    main()
