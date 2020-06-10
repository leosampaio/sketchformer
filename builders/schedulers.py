#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
builders/schedulers.py
Created on Oct 19 2019 16:29

@author: Tu Bui tb0035@surrey.ac.uk
"""

import tensorflow as tf


class WarmupDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    lrate = d**(−0.5) · min(step_num**(−0.5), step_num · warmup_steps**(−1.5))
    """

    def __init__(self, d_model, warmup_steps=4000):
        super(WarmupDecay, self).__init__()

        self.d_model = tf.cast(d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


class StepDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    decay by a factor when certain step is reached
    """

    def __init__(self, init_lr, decay_rate=0.1, decay_steps=50000, min_lr_ratio=1e-2):
        self.init_lr = tf.constant(init_lr, dtype=tf.float32)
        self.decay_steps = decay_steps
        self.decay_rate = tf.constant(decay_rate, tf.float32)
        self.min_lr_ratio = min_lr_ratio

    def __call__(self, step):
        return tf.maximum(
            self.init_lr * tf.math.pow(self.decay_rate, tf.math.floor(step / self.decay_steps)),
            self.init_lr * self.min_lr_ratio)
