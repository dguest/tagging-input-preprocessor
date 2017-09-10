import h5py 
import numpy as np

# Sub selection only happens in the second dimension
def create_dataset(open_file, feature_name, shape):
    try:
        return open_file.create_dataset(feature_name, shape, dtype=np.float32)
    except(RuntimeError):
        return open_file.get(feature_name)


def create_weights(open_file, save_file, set_type):
    feature_name = 'weights/%s' % set_type
    data = open_file.get('jets/%s' % set_type)
    assert data is not None
    N = data.shape[0]
    #save_data = open_file.create_dataset(feature_name, (N, 1), dtype=np.float32)
    save_data = create_dataset(save_file, feature_name, (N, 1))
    sub_selection = [0]
    assert len(sub_selection) == save_data.shape[1]
    copy_in_batches(data, save_data, sub_selection)


def create_high_level_clusters(open_file, save_file, set_type):
    feature_name = 'hl_clusters/%s' % set_type
    data = open_file.get('jets/%s' % set_type)
    assert data is not None
    N = data.shape[0]
    #save_data = open_file.create_dataset(feature_name, (N, 7), dtype=np.float32)
    save_data = create_dataset(save_file, feature_name, (N, 2+7))
    sub_selection = list(range(1,3)) + list(range(4,11))
    assert len(sub_selection) == save_data.shape[1]
    copy_in_batches(data, save_data, sub_selection)

def create_jets(open_file, save_file, set_type):
    feature_name = 'jets/%s' % set_type
    data = open_file.get('jets/%s' % set_type)
    assert data is not None
    N = data.shape[0]
    #save_data = open_file.create_dataset(feature_name, (N, 7), dtype=np.float32)
    save_data = create_dataset(save_file, feature_name, (N, 11))
    sub_selection = range(0,11)
    assert len(sub_selection) == save_data.shape[1]
    copy_in_batches(data, save_data, sub_selection)


def create_high_level_tracks(open_file, save_file, set_type):
    feature_name = 'hl_tracks/%s' % set_type
    data_0 = open_file.get('jets/%s' % set_type)
    data_1 = open_file.get('subjet1/%s' % set_type)
    data_2 = open_file.get('subjet2/%s' % set_type)
    assert data_0 is not None
    assert data_1 is not None
    assert data_2 is not None
    N = data_1.shape[0]
    #save_data = open_file.create_dataset(feature_name, (N, 37*2), dtype=np.float32)
    save_data = create_dataset(save_file, feature_name, (N, 2+37*2))
    sub_selection_0 = range(1,3)
    sub_selection_1 = list(range(0,15)) + list(range(16, 34)) + list(range(36, 40))
    sub_selection_2 = list(range(0,15)) + list(range(16, 34)) + list(range(36, 40))
    assert len(sub_selection_0) + len(sub_selection_1) + len(sub_selection_2) == save_data.shape[1]
    copy_in_batches([data_0, data_1, data_2], save_data, [sub_selection_0, sub_selection_1, sub_selection_2])

def create_mv2c10(open_file, save_file, set_type):
    feature_name = 'mv2c10/%s' % set_type
    data_1 = open_file.get('subjet1/%s' % set_type)
    data_2 = open_file.get('subjet2/%s' % set_type)
    assert data_1 is not None
    assert data_2 is not None
    N = data_1.shape[0]
    #save_data = open_file.create_dataset(feature_name, (N, 2), dtype=np.float32)
    save_data = create_dataset(save_file, feature_name, (N, 2))
    sub_selection_1 = [15]
    sub_selection_2 = [15]
    assert len(sub_selection_1) + len(sub_selection_2) == save_data.shape[1]
    copy_in_batches([data_1, data_2], save_data, [sub_selection_1, sub_selection_2])


def create_mv2c10_extra(open_file, save_file, set_type):
    pass
    print("not implemented")  

def create_high(open_file, save_file, set_type):
    feature_name = 'high/%s' % set_type
    data_1 = open_file.get('hl_clusters/%s' % set_type)
    data_2 = open_file.get('hl_tracks/%s' % set_type)
    assert data_1 is not None
    assert data_2 is not None
    N = data_1.shape[0]
    #save_data = open_file.create_dataset(feature_name, (N, 7+37), dtype=np.float32)
    save_data = create_dataset(save_file, feature_name, (N, 2+7+2+2*37))
    sub_selection_1 = range(0,2+7)
    sub_selection_2 = range(0,2+37*2)
    assert len(sub_selection_1) + len(sub_selection_2) == save_data.shape[1]
    copy_in_batches([data_1, data_2], save_data, [sub_selection_1, sub_selection_2])

def create_low(open_file, save_file, set_type):
    feature_name = 'low/%s' % set_type 
    data_1 = open_file.get('clusters/%s' % set_type)
    data_2 = open_file.get('tracks/%s' % set_type)
    assert data_1 is not None
    assert data_2 is not None
    N = data_1.shape[0]
    #save_data = open_file.create_dataset(feature_name, (N, 5+29), dtype=np.float32)
    save_data = create_dataset(save_file, feature_name, (N, 2+5*60+2+25*60))
    sub_selection_1 = range(0,2+5*60)
    sub_selection_2 = range(0,2+25*60)
    assert len(sub_selection_1) + len(sub_selection_2) == save_data.shape[1]
    copy_in_batches([data_1, data_2], save_data, [sub_selection_1, sub_selection_2])

def create_tracks(open_file, save_file, set_type):
    feature_name = 'tracks/%s' % set_type
    data_1 = open_file.get('jets/%s' % set_type)
    data_2 = open_file.get('tracks/%s' % set_type)
    assert data_1 is not None
    assert data_2 is not None
    N = data_1.shape[0]
    #save_data = open_file.create_dataset(feature_name, (N, 5+29), dtype=np.float32)
    save_data = create_dataset(save_file, feature_name, (N, 2+25*60))
    sub_selection_1 = range(1,3)
    sub_selection_2 = list(range(0, 9)) + list(range(13, 29))
    assert len(sub_selection_1) + len(sub_selection_2) == 2+25
    copy_in_batches([data_1, data_2], save_data, [sub_selection_1, sub_selection_2])

def create_clusters(open_file, save_file, set_type):
    feature_name = 'clusters/%s' % set_type
    data_1 = open_file.get('jets/%s' % set_type)
    data_2 = open_file.get('clusters/%s' % set_type)
    assert data_1 is not None
    assert data_2 is not None
    N = data_1.shape[0]
    #save_data = open_file.create_dataset(feature_name, (N, 5+29), dtype=np.float32)
    save_data = create_dataset(save_file, feature_name, (N, 2+5*60))
    sub_selection_1 = range(1,3)
    sub_selection_2 = range(0,5)
    assert len(sub_selection_1) + len(sub_selection_2) == 2+5
    copy_in_batches([data_1, data_2], save_data, [sub_selection_1, sub_selection_2])

def create_clusters_and_hl_clusters(open_file, save_file, set_type):
    feature_name = 'clusters_and_hl_clusters/%s' % set_type
    data_1 = open_file.get('clusters/%s' % set_type)
    data_2 = open_file.get('hl_clusters/%s' % set_type)
    assert data_1 is not None
    assert data_2 is not None
    N = data_1.shape[0]
    #save_data = open_file.create_dataset(feature_name, (N, 5+29), dtype=np.float32)
    save_data = create_dataset(save_file, feature_name, (N, 2+5*60 + 2+7))
    sub_selection_1 = range(0, 2+5*60)
    sub_selection_2 = range(0, 2+7)
    assert len(sub_selection_1) + len(sub_selection_2) == 2+5*60 + 2+7
    copy_in_batches([data_1, data_2], save_data, [sub_selection_1, sub_selection_2])

def create_tracks_and_hl_tracks(open_file, save_file, set_type):
    feature_name = 'tracks_and_hl_tracks/%s' % set_type
    data_1 = open_file.get('tracks/%s' % set_type)
    data_2 = open_file.get('hl_tracks/%s' % set_type)
    assert data_1 is not None
    assert data_2 is not None
    N = data_1.shape[0]
    #save_data = open_file.create_dataset(feature_name, (N, 5+29), dtype=np.float32)
    save_data = create_dataset(save_file, feature_name, (N, 2+25*60 + 2+2*37))
    sub_selection_1 = range(0,2+25*60)
    sub_selection_2 = range(0, 2+2*37)
    assert len(sub_selection_1) + len(sub_selection_2) == 2+25*60 + 2+2*37
    copy_in_batches([data_1, data_2], save_data, [sub_selection_1, sub_selection_2])

def create_all(open_file, save_file, set_type):
    feature_name = 'all/%s' % set_type
    data_1 = open_file.get('clusters_and_hl_clusters/%s' % set_type)
    data_2 = open_file.get('tracks_and_hl_tracks/%s' % set_type)
    assert data_1 is not None
    assert data_2 is not None
    N = data_1.shape[0]
    #save_data = open_file.create_dataset(feature_name, (N, 5+29), dtype=np.float32)
    save_data = create_dataset(save_file, feature_name, (N, 2+5*60 + 2+7 + 2+25*60 + 2+2*37))
    sub_selection_1 = range(0, 2+5*60 + 2+7)
    sub_selection_2 = range(0, 2+25*60 + 2+2*37)
    assert len(sub_selection_1) + len(sub_selection_2) == 2+5*60 + 2+7 + 2+25*60 + 2+2*37
    copy_in_batches([data_1, data_2], save_data, [sub_selection_1, sub_selection_2])

def flatten_3d_into_2d(data):
    if len(data.shape) == 3:
        reshaped_data = data.reshape(data.shape[0], data.shape[1]*data.shape[2])
        return reshaped_data
    else:
        return data

def sub_select_data(data, start, end, sub_selection):
    if len(data.shape) == 1:
        return data[start:end]
    elif len(data.shape) == 2:
        return data[start:end, sub_selection]
    elif len(data.shape) == 3:
        return data[start:end, sub_selection, :]

def copy_in_batches(data, save_data, sub_selection):
    assert data is not None
    assert save_data is not None

    rounded_num_samples = save_data.shape[0]
    batch_size = 100
    
    if data is list:
        assert len(data) == len(sub_selection), "a sub_selection must be specified for each dataset"

    for start, end in zip(range(0, rounded_num_samples, batch_size), 
                          range(batch_size, rounded_num_samples+batch_size, batch_size)):
        if data.__class__ is list:
            assert len(data) == len(sub_selection), "a sub_selection must be specified for each dataset"
            if len(data) == 2:
                data_1 = data[0]
                data_2 = data[1]
                sub_selection_1 = sub_selection[0]
                sub_selection_2 = sub_selection[1]
                
                temp_data_1 = sub_select_data(data_1, start, end, sub_selection_1)
                temp_data_2 = sub_select_data(data_2, start, end, sub_selection_2)

                temp_data_1 = flatten_3d_into_2d(temp_data_1)
                temp_data_2 = flatten_3d_into_2d(temp_data_2)

                mini_batch = np.hstack((temp_data_1, temp_data_2))

            elif len(data) == 3:
                data_1 = data[0]
                data_2 = data[1]
                data_3 = data[2]
                sub_selection_1 = sub_selection[0]
                sub_selection_2 = sub_selection[1]
                sub_selection_3 = sub_selection[2]
                temp_data_1 = sub_select_data(data_1, start, end, sub_selection_1)
                temp_data_2 = sub_select_data(data_2, start, end, sub_selection_2)
                temp_data_3 = sub_select_data(data_3, start, end, sub_selection_3)
                temp_data_1 = flatten_3d_into_2d(temp_data_1)
                temp_data_2 = flatten_3d_into_2d(temp_data_2)
                temp_data_3 = flatten_3d_into_2d(temp_data_3)
                merged_1_2 = np.hstack((temp_data_1, temp_data_2))
                mini_batch = np.hstack((merged_1_2, temp_data_3))

        else:
            mini_batch = sub_select_data(data, start, end, sub_selection)
            mini_batch = flatten_3d_into_2d(mini_batch)
                
        num_dims = len(mini_batch.shape)

        if num_dims == 1:
            save_data[start:end] = mini_batch
        elif num_dims == 2:
            save_data[start:end, :] = mini_batch
        elif num_dims == 3:
            save_data[start:end, :, :] = mini_batch

if __name__ == "__main__":
    file_path = "./"
    load_name = "flattened_data.h5"
    save_name = "test_data.h5"
    hf = h5py.File(file_path + load_name, 'r')
    save_file = h5py.File(file_path + save_name, 'a')
    print(hf.keys())
    
    for set_type in ['',]:
        print(set_type)
        print("Splitting weights")
        create_weights(hf, save_file, set_type)
        print("Splitting hl clusters")
        create_high_level_clusters(hf, save_file, set_type)
        print("splitting hl tracks")
        create_high_level_tracks(hf, save_file, set_type)
        print("splitting mv2c10")
        create_mv2c10(hf, save_file, set_type)
        print("splitting tracks")
        create_tracks(hf, save_file, set_type)
        print("splitting clusters")
        create_clusters(hf, save_file, set_type)
        print("splitting high")
        create_high(save_file, save_file, set_type)
        print("splitting low")
        create_low(save_file, save_file, set_type)
        print("splitting clusters_and_hl_clusters")
        create_clusters_and_hl_clusters(save_file, save_file, set_type)
        print("splitting tracks_and_hl_tracks")
        create_tracks_and_hl_tracks(save_file, save_file, set_type)
        print("splitting all")
        create_all(save_file, save_file, set_type)
        print("creating jets")
        create_jets(hf, save_file, set_type)

