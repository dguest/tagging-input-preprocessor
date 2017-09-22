from __future__ import print_function

import numpy as np
import h5py

def flatten(ds):
    """
    Flattens a named numpy array so it can be used with pure numpy.
    Input: named array
    Output: numpy array

    Example;
    print(flatten(sample_jets[['pt', 'eta']]))
    """
    ftype = [(n, float) for n in ds.dtype.names]
    flat = ds.astype(ftype).view(float).reshape(ds.shape + (-1,))
    swapped = flat.swapaxes(1, len(ds.shape)) # swapping is done to arrange array due to new dimension
    return swapped

def shuffle_samples(data, axis=0):
    """ 
    Shuffles the samples (entries) in a numpy array along the given axis
    Input: numpy array
    Output: shuffled numpy array

    Example
    shuffled_data = shuffle_samples(data)

    Possible improvements:
    in place shuffling
    generalize for more than 3 dimensions
    generalize for any axis (maybe we can do both of this by using string execution)
    """

    assert axis==0, "not yet implemented for other axis"
    num_samples = data.shape[axis]
    num_dimensions = len(data.shape)
    indices = range(num_samples)
    np.random.shuffle(indices)

    if num_dimensions == 1:
        return data[indices]
    elif num_dimensions == 2:
        return data[indices, :]
    elif num_dimensions == 3:
        return data[indices, :, :]
    else:
        assert 1==0, "not implemented but easy to implement"


def count_num_samples_from_hdf5(file_name, round_down=1.0):
    hf = h5py.File(file_name, 'r')
    feature_names = hf.keys()
    num_samples = None
    for feature_name in feature_names:
        data = hf.get(feature_name)
        assert data is not None, "%s, %s"%(file_name, feature_name)
        if num_samples is None:
            num_samples = data.shape[0]
            num_samples = np.floor(num_samples/round_down)
        else:
            assert num_samples == np.floor(data.shape[0]/round_down), "num samples changed for %s  %s"%(file_name, feature_name)
    return num_samples


def count_num_samples_from_hdf5_file_list(list_of_files, round_down=1.0):
    total_num_samples = 0
    for file_name in list_of_files:
        num_samples = count_num_samples_from_hdf5(file_name, round_down)
        total_num_samples += num_samples
    return total_num_samples


def get_size_of_features(file_name):
    pass

def reshape_to_flat(data):
    assert len(data.shape) == 3
    return data.reshape((data.shape[0], data.shape[1] * data.shape[2]))






def test_shuffle_samples():
    # 1 dimension
    np.random.seed(2)
    a = np.asarray(range(4))
    b = shuffle_samples(a)
    c = a[[2, 3, 1, 0]]
    assert np.all(np.equal(b, c))

    np.random.seed(2)
    # 2 dimensions
    a = np.asarray(range(12)).reshape((4,3))
    b = shuffle_samples(a)
    c = a[[2, 3, 1, 0], :]
    assert np.all(np.equal(b, c))

    # 3 dimensions
    np.random.seed(2)
    a = np.asarray(range(24)).reshape((4,3,2))
    b = shuffle_samples(a)
    c = a[[2, 3, 1, 0], :, :]
    assert np.all(np.equal(b, c))

if __name__ == "__main__":
    test_shuffle_samples()

