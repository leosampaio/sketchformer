#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
builders/keras_metrics.py
Created on Oct 19 2019 10:11

@author: Tu Bui tb0035@surrey.ac.uk
"""

import tensorflow as tf


class MetricManager(object):

    def __init__(self):
        self.metric_names = []
        self.metric_fns = {}

    def add_mean_metric(self, name):
        self.metric_names.append(name)
        self.metric_fns[name] = tf.keras.metrics.Mean(name='{}_metric'.format(name))

    def add_sparse_categorical_accuracy(self, name):
        self.metric_names.append(name)
        self.metric_fns[name] = tf.keras.metrics.SparseCategoricalAccuracy(name='{}_metric'.format(name))

    def compute(self, name, *args):
        assert name in self.metric_names, 'Error! {} metric not found.'.format(name)
        self.metric_fns[name](*args)

    def reset(self):
        for metric_fns in self.metric_fns.values():
            try:
                metric_fns.reset_states()
            except AttributeError as _:
                pass  # this metric is not a keras Metric object

    def get_results(self):
        return [metric_fns.result() for metric_fns in self.metric_fns]

    def get_results_as_dict(self):
        return {m: self.metric_fns[m].result().numpy() for m in self.metric_names}
