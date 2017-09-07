import numpy as np
import h5py
import utils

def create_dataset(open_file, feature_name, shape):
    try:
        return open_file.create_dataset(feature_name, shape, dtype=np.float32)
    except(RuntimeError):
        return open_file.get(feature_name)

# Merges many hdf5 files into one
data_label = 'signal'
new_file_dataset_name = "small_test_data_%s_.h5" % data_label
path = "/baldig/physicsprojects/atlas/hbb/raw_data/v_3/"
file_list_s = ['d301488_j1.h5',]
#file_list_s = ['d301488_j1.h5', 'd301489_j2.h5', 'd301490_j3.h5', 'd301491_j4.h5',
#               'd301492_j5.h5', 'd301493_j6.h5', 'd301494_j7.h5', 'd301495_j8.h5',
#               'd301496_j9.h5', 'd301497_j10.h5', 'd301498_j11.h5', 'd301499_j12.h5',
#               'd301500_j13.h5', 'd301501_j14.h5', 'd301502_j15.h5', 'd301503_j16.h5',
#               'd301504_j17.h5', 'd301505_j18.h5', 'd301506_j19.h5', 'd301507_j20.h5']
file_list_bg = [#'d361021_j27.h5', 
                'd361022_j28.h5', 'd361023_j29.h5','d361024_j30.h5', 
                'd361025_j31.h5', 'd361026_j32.h5', 'd361027_j33.h5', 'd361028_j34.h5',
                'd361029_j35.h5', 'd361030_j36.h5', 'd361031_j37.h5', 'd361032_j38.h5']

if data_label == 'signal':
    file_list = file_list_s
elif data_label == 'bg':
    file_list = file_list_bg

f_names = []
for name in file_list:
    f_names.append(path + name)

feature_names = [u'clusters', u'jets', u'subjet1', u'subjet2', u'tracks']

files = []
for f_name in f_names:
    files.append(h5py.File(f_name, 'r'))

# Calculate total samples
round_down = 1.0
total_samples = utils.count_num_samples_from_hdf5_file_list(f_names, round_down)
print "total samples", total_samples

new_hdf5 = h5py.File(path + new_file_dataset_name, 'w')
for feature_name in feature_names:
    print feature_name
    start = 0
    end = 0
    data = None
    old_names = None
    for f_name in f_names:
        print "loading %s" % f_name
        f = h5py.File(f_name)
        data = f.get(feature_name)
        col_names = data.dtype.names
        assert data is not None
        N = int(total_samples)
        new_sizes = {u'clusters': (N, 5, 60),
                 u'jets': (N, 11),
                 u'subjet1': (N, 40),
                 u'subjet2': (N, 40),
                 u'subjet3': (N, 40),
                 u'tracks': (N, 29, 60),
                }

        # Create the new empty dataset
        if start == 0:
            create_dataset(new_hdf5, feature_name, new_sizes[feature_name])
         
        end = end + int(np.floor(data.shape[0]/round_down))

        save_data = new_hdf5.get(feature_name)
        # this could be made smaller to acomodate RAM requirements
        #data = utils.flatten(data)
        num_samples_this_file = int(np.floor(data.shape[0]/round_down))
        print start, end, data.shape[0], end-start
        if old_names is None:
            old_names = col_names
        assert old_names == col_names, old_names + col_names
        data = data[col_names] 
        if len(data.shape) == 1:
            data = data[:]
        elif len(data.shape) == 2:
            data = data[:, :]
        elif len(data.shape) == 3:
            data = data[:, :, :]   

        #data = data[list(col_names)]
        assert data is not None
        if len(data.shape) == 1:
            data = utils.flatten(data[0:num_samples_this_file])
            save_data[start:end] = data
        elif len(data.shape) == 2:
            data = utils.flatten(data[0:num_samples_this_file, :])
            save_data[start:end, :] = data
        elif len(data.shape) == 3:
            data = utils.flatten(data[0:num_samples_this_file, :, :])
            save_data[start:end, :, :] = data

        start = start + num_samples_this_file




