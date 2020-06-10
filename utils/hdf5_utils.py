#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hdf5_utils.py
Created on Aug 16 2019 18:12
Classes for read and write hdf5 dataset

@author: Tu Bui tb0035@surrey.ac.uk
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import h5py as h5
import numpy as np


class HDF5Write(object):
    """
    write data into hdf5 (.h5) file
    Example usage:
    data = HDF5Write('mydata.h5', ['images', 'labels'], [(256,256,3), (1,)],
                    [np.float32, np.int64], [128, None], "gzip")
    data.append('images', np.random.rand(10, 256, 256, 3))
    data.append('labels', np.int64(range(10)))
    """
    def __init__(self, data_path, dataset_names, shapes, dtypes=np.float32,
                 chunk_lens=1, compression="gzip"):
        """
        initialisation
        :param data_path: path to dataset, extension .h5 or .hdf5
        :param dataset_names: list of dataset names
        :param shapes: corresponding list of (tuple) shapes
        :param dtypes: corresponding list of dtypes
        :param chunk_lens: corresponding list of chunk length (None if don't want to be chunked)
        :param compression: compression
        """
        self.data_path = data_path
        self.dataset_names = [dataset_names, ] if isinstance(dataset_names, str) else dataset_names
        self.shapes = [shapes, ] if isinstance(shapes, tuple) else shapes
        self.dtypes = dtypes if isinstance(dtypes, list) else [dtypes, ]
        self.compression = compression
        self.chunk_lens = [chunk_lens, ] if isinstance(chunk_lens, int) else chunk_lens
        self.ids = {}
        with h5.File(self.data_path, 'w') as f:
            for i in range(len(self.dataset_names)):
                f.create_dataset(self.dataset_names[i],
                                 shape=(0, ) + self.shapes[i],
                                 maxshape=(None, ) + self.shapes[i],
                                 dtype=self.dtypes[i],
                                 chunks=(self.chunk_lens[i],) + self.shapes[i],
                                 compression=self.compression)
                self.ids[self.dataset_names[i]] = 0  # store index of corresponding dataset

    def append(self, name, data):
        """
        append some data into hdf5 dataset
        :param name: name of the dataset
        :param data: data values
        :return: None
        """
        assert name in self.dataset_names, 'Error! %s not in %s' % (name, self.dataset_names)
        name_id = self.dataset_names.index(name)
        n = len(data)
        with h5.File(self.data_path, 'a') as f:
            dset = f[name]
            dset.resize((self.ids[name] + n,) + self.shapes[name_id])
            dset[self.ids[name]:] = data
            self.ids[name] += n
            f.flush()


class HDF5Read(object):
    """
    Class to read hdf5 dataset created by HDF5Write
    Usage:
    dset = HDF5Read('my_dataset.h5')
    img, label = dset.get_datum(0)
    imgs, labels = dset.get_data([0, 2, 5])
    imgs, labels = dset.get_seq_data(range(3))
    """
    def __init__(self, data_path, load_all=False):
        """
        initializer
        :param data_path: path to h5/hdf5 data
        :param load_all: if True load the whole database into memory
        """
        self.f = h5.File(data_path, 'r')
        if load_all:
            self.labels = self.f['labels'][...]
            self.images = self.f['images'][...]
            self.ids = self.f['ids'][...]
        else:
            self.labels = self.f['labels']
            self.images = self.f['images']
            self.ids = self.f['ids']

    def __del__(self):
        try:
            self.f.close()
        except Exception as e:
            pass

    def get_size(self):
        """get number of samples in this dataset"""
        return len(self.labels)

    def get_datum(self, ind):
        """
        return image given its index
        :param ind: index of the image [0, N)
        :return: image associated with the ind and label
        """
        start, end = self.ids[ind]
        datum = self.images[start:end, ...]
        label = self.labels[ind, ...]
        return datum, label

    def get_data(self, indices):
        """
        return data in batches
        :param indices: list of index
        :return: array of images, array of labels
        """
        labels = self.labels[indices, ...]
        data = []
        for ind in indices:
            start, end = self.ids[ind]
            datum = self.images[start:end, ...]
            data.append(datum)
        return np.array(data), labels

    def get_seq_data(self, start, end):
        """
        special case of get_data where indices is a continous squence with start and end id
        :param start: start index
        :param end: end index
        :return: array of images, array of labels
        """
        assert end > start + 1, 'Error! '
        labels = self.labels[start:end, ...]
        idbox = self.ids[start:end, ...]
        chunk = self.images[idbox[0, 0]: idbox[-1, 1], ...]
        slices = idbox[1:, 0] - idbox[0, 0]
        return np.split(chunk, slices, axis=0), labels
