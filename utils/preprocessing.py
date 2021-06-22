import tensorflow as tf
import numpy as np
import h5py


class LoadDataset:

    @staticmethod
    def target_dataset():
        path = "C:\\Users\\khodabakhshandeh\\Desktop\\Projects\\few-shot\\dataset\\usps.h5"
        with h5py.File(path, 'r') as hf:
            train = hf.get('train')
            x_tr = train.get('data')[:]
            x_tr = np.reshape(x_tr, (x_tr.shape[0], 16, 16))
            x_tr = np.expand_dims(x_tr, axis=-1)
            y_tr = train.get('target')[:]
            test = hf.get('test')
            x_te = test.get('data')[:]
            x_te = np.reshape(x_te, (x_te.shape[0], 16, 16))
            x_te = np.expand_dims(x_te, axis=-1)
            y_te = test.get('target')[:]
        return x_tr, y_tr, x_te, y_te

    @staticmethod
    def resize_images(data):
        data = tf.image.resize(data, [16, 16], method="bilinear", antialias=True)
        return data

    @staticmethod
    def normalize(data):
        min_vals = np.amin(data, axis=(1, 2))
        max_vals = np.amax(data, axis=(1, 2))
        var = max_vals - min_vals
        data = np.subtract(data, min_vals[..., np.newaxis, np.newaxis])
        data = np.true_divide(data, var[..., np.newaxis, np.newaxis])
        data = np.expand_dims(data, axis=-1)
        data = LoadDataset.resize_images(data).numpy()
        return data

