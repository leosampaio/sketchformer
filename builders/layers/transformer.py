# -*- coding: utf-8 -*-
"""
builders/layers/transformer.py
Created on 01/05/19
@author: Tu Bui tb00083@surrey.ac.uk
"""

import tensorflow as tf
from tensorflow.keras import backend as K
from ..utils import scaled_dot_product_attention, positional_encoding


class SelfAttnV1(tf.keras.layers.Layer):
    """
    Keras attention layer for a sequence
    learn weight for each time step
    This implementation uses the attention formula proposed by  Sukhbaatar etal. 2015
    https://papers.nips.cc/paper/5846-end-to-end-memory-networks.pdf

    Example:
        from tensorflow.keras.layers import Input, LSTM
        from attn_rnn import AttnRNN

        input_data = Input(shape=(32,128))  # [?, 32, 128]
        x = LSTM(10, return_sequences=True)(input_data)  # [?, 32, 10]
        x, w = SelfAttn()(x)  # x: [?, 10], w: [?, 32]

        where w is the attention weight for each time step (useful for visualisation/evaluation)
    """

    def __init__(self, units=None, **kwargs):
        """
        Layer initialisation
        :param units: define the embedding dimension. If not specified (default),
                      it will be set to feat dimension.
        :param kwargs:
        """
        self.units = units
        super(SelfAttnV1, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        fdim = input_shape[-1]
        if self.units is None:
            self.units = fdim

        self.W = self.add_weight(name='W_attn',
                                 shape=(fdim, self.units),
                                 initializer='normal',
                                 trainable=True)
        self.b = self.add_weight(name='b_attn',
                                 shape=(self.units,),
                                 initializer='zeros',
                                 trainable=True)
        self.V = self.add_weight(name='V_attn',
                                 shape=(self.units, 1),
                                 initializer='uniform',
                                 trainable=True)
        super(SelfAttnV1, self).build(input_shape)

    def call(self, x):
        """
        ui = tanh(xW+b)
        a = softmax(uV)
        o = sum(a*x)
        :param x: input tensor [batch_size, time_step, feat_len]
        :return: output tensor [batch_size, feat_len]
        """
        # ui = tanh(xW+b)
        ui = K.tanh(K.bias_add(K.dot(x, self.W), self.b))  # [B, T, L]
        # a = softmax(uV)
        ai = K.softmax(K.dot(ui, self.V), axis=1)  # [B, T, 1]
        o = K.sum(x * ai, axis=1, keepdims=False)
        return o, ai

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.units


class SelfAttnV2(tf.keras.layers.Layer):
    """
    Version2 of selfattn
    if units is not None: add a dense layer after the attention to change output dimension
    """

    def __init__(self, units=None, **kwargs):
        """
        Layer initialisation
        :param units: define the embedding dimension. If not specified (default),
                      it will be set to feat dimension.
        :param kwargs:
        """
        self.units = units
        super(SelfAttnV2, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        fdim = input_shape[-1]
        if self.units:
            self.embeding_layer = tf.keras.layers.Dense(self.units)

        self.W = self.add_weight(name='W_attn',
                                 shape=(fdim, fdim),
                                 initializer='normal',
                                 trainable=True)
        self.b = self.add_weight(name='b_attn',
                                 shape=(fdim,),
                                 initializer='zeros',
                                 trainable=True)
        self.V = self.add_weight(name='V_attn',
                                 shape=(fdim, 1),
                                 initializer='uniform',
                                 trainable=True)
        super(SelfAttnV2, self).build(input_shape)

    def call(self, x):
        """
        ui = tanh(xW+b)
        a = softmax(uV)
        o = sum(a*x)
        :param x: input tensor [batch_size, time_step, feat_len]
        :return: output tensor [batch_size, new_feat_len]
        """
        # ui = tanh(xW+b)
        ui = K.tanh(K.bias_add(K.dot(x, self.W), self.b))  # [B, T, L]
        # a = softmax(uV)
        ai = K.softmax(K.dot(ui, self.V), axis=1)  # [B, T, 1]
        o = K.sum(x * ai, axis=1, keepdims=False)  # [B, T, L]
        if self.units:
            o = self.embeding_layer(o)
        return o, ai

    def compute_output_shape(self, input_shape):
        if self.units:
            return input_shape[0], self.units
        else:
            return input_shape[0], input_shape[-1]


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    multi head attention for transformer
    """

    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


class EncoderLayer(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


class DecoderLayer(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2


class Encoder(tf.keras.layers.Layer):

    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding=1000, rate=0.1,
                 use_continuous_input=False):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        if use_continuous_input:
            self.embedding = tf.keras.layers.Dense(d_model)
        else:
            self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)

        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, ...]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


class Decoder(tf.keras.layers.Layer):

    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 maximum_position_encoding=1000, rate=0.1,
                 use_continuous_input=False):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        if use_continuous_input:
            self.embedding = tf.keras.layers.Dense(d_model)
        else:
            self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)

        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, ...]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                   look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights


class DenseExpander(tf.keras.layers.Layer):
    """
    Expand tensor using Dense conv
    input: (batch_size, feat_dim_in)
    output: (batch_size, seq_len, feat_dim_out)
    """

    def __init__(self, seq_len, feat_dim_out=0):
        super(DenseExpander, self).__init__()
        self.seq_len = seq_len
        self.feat_dim_out = feat_dim_out

    def build(self, input_shape):
        assert len(input_shape) == 2, 'Error! input tensor must be 2D'
        if self.feat_dim_out:
            self.project_layer = tf.keras.layers.Dense(self.feat_dim_out, activation='relu')
        self.expand_layer = tf.keras.layers.Dense(self.seq_len)
        super(DenseExpander, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        feat_dim_out = self.feat_dim_out if self.feat_dim_out else input_shape[-1]
        return input_shape[0], self.seq_len, feat_dim_out

    def call(self, x):
        if self.feat_dim_out:
            x = self.project_layer(x)
        x = tf.expand_dims(x, axis=2)  # (batch_size, feat_dim_out, 1)
        x = self.expand_layer(x)  # (batch_size, d_model, seq_len)
        x = tf.transpose(x, perm=[0, 2, 1])  # (batch_size, seq_len, feat_dim_out)
        return x
