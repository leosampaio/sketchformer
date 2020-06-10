"""
builders/losses.py
Created on Oct 16 2019 17:34
"""

import tensorflow as tf


class LossManager(object):

    def __init__(self):
        self.loss_names = []
        self.loss_fns = {}
        self.loss_weights = {}

    def _add_loss(self, name, weight, func):
        self.loss_names.append(name)
        self.loss_weights[name] = weight
        self.loss_fns[name] = func

    def add_sparse_categorical_crossentropy(self, name='class', weight=1.0):
        self._add_loss(
            name, weight,
            tf.keras.losses.SparseCategoricalCrossentropy(name=name + '_loss'))

    def add_reconstruction_loss(self, name='recon', weight=1.0):
        loss_recon_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none', name=name + '_loss')

        def loss_function(real, pred):
            real = tf.squeeze(real)
            pred = tf.squeeze(pred)
            mask = tf.math.logical_not(tf.math.equal(real, 0))
            loss_ = loss_recon_object(real, pred)

            mask = tf.cast(mask, dtype=loss_.dtype)
            loss_ *= mask

            return tf.reduce_mean(loss_)

        self._add_loss(name, weight, loss_function)

    def add_continuous_reconstruction_loss(self, name='recon', weight=1.0):

        def loss_function(real, pred):
            mask = tf.math.logical_not(tf.math.equal(real[..., -1], 1))

            pred_locations = pred[:, :, :2]
            pred_metadata = pred[:, :, 2:]
            tgt_locations = real[:, :, :2]
            tgt_metadata = real[:, :, 2:]
            location_loss = tf.losses.mean_squared_error(tgt_locations,
                                                         pred_locations)
            metadata_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.argmax(tgt_metadata, axis=-1),
                logits=pred_metadata)
            metadata_loss = tf.reduce_mean(metadata_loss)

            loss_ = location_loss + metadata_loss

            mask = tf.cast(mask, dtype=loss_.dtype)
            loss_ *= mask

            return tf.reduce_mean(loss_)

        self._add_loss(name, weight, loss_function)

    def add_mae_loss(self, name, weight=1.):
        self._add_loss(name, weight, tf.keras.losses.MAE)

    def add_mean_loss(self, name, weight=1.):
        self._add_loss(name, weight, tf.reduce_mean)

    def add_mse_loss(self, name, weight=1.):
        self._add_loss(name, weight, tf.keras.losses.MSE)

    def compute_all_loss(self, rp_dict):
        losses = {}
        for lid, name in enumerate(self.loss_names):
            losses[name] = self.loss_weights[name] * self.loss_fns[name](*rp_dict[name])
        return losses

    def compute_loss(self, name, *args):
        assert name in self.loss_names, "Error! Loss name {} not found".format(name)
        return self.loss_weights[name] * self.loss_fns[name](*args)
