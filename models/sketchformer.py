"""
sketchformer.py
Created on Oct 06 2019 16:34
ref: https://www.tensorflow.org/tutorials/text/transformer
"""

import tensorflow as tf
import numpy as np

import builders
from utils.hparams import HParams
from core.models import BaseModel
from .evaluation_mixin import TransformerMetricsMixin
from builders.layers.transformer import (Encoder, Decoder, DenseExpander)


class Transformer(BaseModel, TransformerMetricsMixin):
    name = 'sketch-transformer-tf2'
    quick_metrics = ['recon_loss', 'recon_acc', 'class_loss', 'class_acc',
                     'total_loss']
    slow_metrics = ["sketch-reconstruction", "val-clas-acc", "tsne",
                    "tsne-predicted"]

    @classmethod
    def specific_default_hparams(cls):
        """Return default HParams"""
        hparams = HParams(
            num_layers=4,  # number of Multihead Attn + ffn blocks
            d_model=128,  # internal model dimension
            dff=512,
            num_heads=8,
            dropout_rate=0.1,

            lowerdim=256,
            attn_version=1,

            do_classification=True,  # softmax classification
            class_weight=1.0,
            class_buffer_layers=0,  # buffer FC layers before classifier
            class_dropout=0.1,  # dropout rate for each class buffer layer

            do_reconstruction=True,  # reconstruction
            recon_weight=1.0,
            blind_decoder_mask=True,  # if True, the decoder knows padding location of the input

            # training params
            is_training=True,
            optimizer='Adam',  # SGD, Adam, sgd, adam
            lr=0.01,  # initial lr
            lr_scheduler='WarmupDecay',  # defined in core.lr_scheduler_tf20
            warmup_steps=10000,
        )
        return hparams

    def __init__(self, hps, dataset, out_dir, experiment_id):
        self.losses_manager = builders.losses.LossManager()
        self.metrics_manager = builders.keras_metrics.MetricManager()

        self.vocab_size = dataset.tokenizer.VOCAB_SIZE if not dataset.hps['use_continuous_data'] else None
        self.seq_len = dataset.hps['max_seq_len']
        super().__init__(hps, dataset, out_dir, experiment_id)

    def build_model(self):

        if self.hps['attn_version'] == 1:
            SelfAttn = getattr(builders.layers.transformer, 'SelfAttnV1')
        else:
            SelfAttn = getattr(builders.layers.transformer, 'SelfAttnV2')

        self.encoder = Encoder(
            num_layers=self.hps['num_layers'],
            d_model=self.hps['d_model'],
            num_heads=self.hps['num_heads'], dff=self.hps['dff'],
            input_vocab_size=self.vocab_size, rate=self.hps['dropout_rate'],
            use_continuous_input=self.dataset.hps['use_continuous_data'])
        if self.hps['do_reconstruction']:
            self.decoder = Decoder(
                num_layers=self.hps['num_layers'],
                d_model=self.hps['d_model'],
                num_heads=self.hps['num_heads'], dff=self.hps['dff'],
                target_vocab_size=self.vocab_size,
                rate=self.hps['dropout_rate'],
                use_continuous_input=self.dataset.hps['use_continuous_data'])

            if self.dataset.hps['use_continuous_data']:
                self.output_layer = tf.keras.layers.Dense(5)  # go back to original space
                self.losses_manager.add_continuous_reconstruction_loss('recon',
                                                                       weight=self.hps['recon_weight'])
                self.metrics_manager.add_mean_metric('recon_loss')
            else:
                self.output_layer = tf.keras.layers.Dense(self.vocab_size)
                self.losses_manager.add_reconstruction_loss('recon', weight=self.hps['recon_weight'])
                self.metrics_manager.add_mean_metric('recon_loss')
                self.metrics_manager.add_sparse_categorical_accuracy('recon_acc')

        if self.hps['lowerdim']:
            self.bottleneck_layer = SelfAttn(self.hps['lowerdim'])
            self.expand_layer = DenseExpander(self.seq_len)  # for reconstruction
            if self.hps['do_classification']:
                self.classify_layer = tf.keras.layers.Dense(self.dataset.n_classes, activation='softmax')
                self.class_buffer = [tf.keras.layers.Dense(self.hps['lowerdim'], activation='relu') for _ in
                                     range(self.hps['class_buffer_layers'])]
                self.class_dropout = [tf.keras.layers.Dropout(self.hps['class_dropout']) for _ in
                                      range(self.hps['class_buffer_layers'])]
                self.losses_manager.add_sparse_categorical_crossentropy('class', weight=self.hps['class_weight'])
                self.metrics_manager.add_mean_metric('class_loss')
                self.metrics_manager.add_sparse_categorical_accuracy('class_acc')

        self.metrics_manager.add_mean_metric('total_loss')

        # prepare learning rate schedulers
        if self.hps['lr_scheduler'] == 'WarmupDecay' or self.hps['lr_scheduler'] == 'warmup-decay':
            self.learning_rate = builders.schedulers.WarmupDecay(
                self.hps['d_model'], warmup_steps=5000)
        elif self.hps['lr_scheduler'] == 'step-decay':  # stepdecay
            self.learning_rate = builders.schedulers.StepDecay(
                self.hps['lr'], decay_steps=5000,
                decay_rate=0.5, min_lr=1e-2)

        # prepare optimizers
        if self.hps['optimizer'].lower() == 'adam':
            self.optimizer = tf.keras.optimizers.Adam(
                self.learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        elif self.hps['optimizer'].lower() == 'sgd':
            self.optimizer = tf.keras.optimizers.SGD(
                self.learning_rate, momentum=0.9)

        # finally, build the model trainer
        self.prepare_model_trainer()

    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        """Gather model outputs required in loss computation
        Returns a dictionary containing the outputs with shapes:
            embedding   -- (batch_size, lowerdim)
            recon       -- (batch_size, seq_len, d_input)
            class       -- (batch_size, 1)
            z_mean      -- (batch_size, lowerdim)
            z_log_var   -- (batch_size, lowerdim)
        """
        out_keys = ['embedding', 'recon', 'class', 'z_mean', 'z_log_var']
        enc_outputs = self.encode(inp, enc_padding_mask, training)
        out = {key: enc_outputs[key] for key in out_keys if key in enc_outputs}
        if self.hps['do_reconstruction']:
            dec_outputs = self.decode(enc_outputs['embedding'], tar, dec_padding_mask, look_ahead_mask, training)
            out.update({key: dec_outputs[key] for key in out_keys if key in dec_outputs})

        return out

    def encode(self, inp, inp_mask, training):
        """out_keys = [enc_output, embedding, class, z_mean, z_log_var]"""
        out = {'enc_output': self.encoder(inp, training, inp_mask), 'class': None}
        if self.hps['lowerdim']:
            bottle_neck, _ = self.bottleneck_layer(out['enc_output'])
            out['embedding'] = bottle_neck
            if self.hps['do_classification']:
                out['class'] = self.classify_from_embedding(out['embedding'], training)
        else:
            out['embedding'] = out['enc_output']

        return out

    def encode_from_seq(self, inp_seq):
        """same as encode but compute mask inside. Useful for test"""
        dtype = tf.float32 if self.dataset.hps['use_continuous_data'] else tf.int64
        encoder_input = tf.cast(np.array(inp_seq) + np.zeros((1, 1)), dtype)  # why?
        enc_padding_mask = builders.utils.create_padding_mask(encoder_input)
        res = self.encode(encoder_input, enc_padding_mask, training=False)
        return res

    def decode(self, embedding, target, target_mask, look_ahead_mask, training):
        """Generate logits for each value in the target sequence."""
        padding_mask = tf.zeros_like(target_mask) if self.hps['blind_decoder_mask'] else target_mask
        if self.hps['lowerdim']:  # need to expand embedding
            pre_decoder = self.expand_layer(embedding)
        else:
            pre_decoder = embedding
        dec_output, attention_weights = self.decoder(
            target, pre_decoder, training, look_ahead_mask, padding_mask)
        final_output = self.output_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
        out = {'recon': final_output, 'attn_weights': attention_weights}
        return out

    def classify_from_embedding(self, embedding, training):
        """Implement classification branch
        :param embedding: bottleneck layer
        :param training: training or test (affect dropout)
        :return: N-way logits
        """
        if self.hps['class_buffer_layers']:
            for lid in range(self.hps['class_buffer_layers']):
                if lid == 0:
                    fc = self.class_buffer[lid](embedding)  # keep embedding intact
                else:
                    fc = self.class_buffer[lid](fc)
                fc = self.class_dropout[lid](fc, training=training)
        else:
            fc = embedding
        pred_labels = self.classify_layer(fc)
        return pred_labels

    def predict(self, inp_seq):
        """Forward pass through the model at test mode.
        :param inp_seq: sketch as a sequence of integers. (list, numpy 1D array)
        :return: (reconstructed, class labels, attn_weights, ...)
        """
        out = self.encode_from_seq(inp_seq)
        if self.hps['do_classification']:
            pred_labels = tf.cast(tf.argmax(out['class'], axis=-1), tf.int32)
            out['class'] = pred_labels
        if self.hps['do_reconstruction']:
            if self.hps['blind_decoder_mask']:
                tlen = None
            else:
                if self.dataset.hps['use_continuous_data']:
                    tlen = tf.reduce_sum(tf.cast(inp_seq[..., -1] != 1, tf.float32), axis=-1)
                else:
                    tlen = tf.reduce_sum(tf.cast(inp_seq > 0, tf.float32), axis=-1)  # exact len
            dec_output = self.predict_from_embedding(out['embedding'], tlen)
            out['recon'] = dec_output['recon']
            out['attn_weights'] = dec_output['attn_weights']
        return out

    def predict_class(self, inp_seq):
        out = self.encode_from_seq(inp_seq)
        if self.hps['do_classification']:
            pred_labels = tf.cast(tf.argmax(out['class'], axis=-1), tf.int32)
            out['class'] = pred_labels
        return out

    def make_dummy_input(self, expected_len, nattn, batch_size):
        nignore = self.seq_len - nattn

        if self.dataset.hps['use_continuous_data']:
            dummy = tf.concat((
                tf.ones((batch_size, nattn, 5)) * [0., 0., 0., 0., 0.],
                tf.ones((batch_size, nignore, 5)) * [0., 0., 0., 0., 1.]
            ), axis=1)
        else:
            if expected_len is None:
                dummy = tf.concat((
                    tf.ones((batch_size, nattn)) * [1],
                    tf.ones((batch_size, nignore)) * [0]
                ), axis=1)
            else:
                dummy = []
                for i_nattn, i_nignore in zip(nattn, nignore):
                    dummy_slice = tf.concat((
                        tf.ones((1, i_nattn)) * [1],
                        tf.ones((1, i_nignore)) * [0]
                    ), axis=1)
                    dummy.append(dummy_slice)
                dummy = tf.concat(dummy, axis=0)
        return dummy

    def predict_from_embedding(self, emb, expected_len=None):
        """
        reconstruct sketch from bottle neck layer
        different from predict: input is embedding instead of a sketch
        :param emb: embedding vector
        :param expected_len: expected length as if input sketch is known (will be ignored if blind_decoder_mask=True
        :return: dict of relevant outputs
        """
        out = {}
        embedding = tf.cast(emb + np.zeros((1, 1)), dtype=tf.float32)  # why?
        if self.hps['do_reconstruction']:
            if self.dataset.hps['use_continuous_data']:
                decoder_input = [0., 0., 1., 0., 0.]
                output = tf.ones((emb.shape[0], 1, 5)) * decoder_input
            else:
                decoder_input = [self.dataset.tokenizer.SOS]
                output = tf.cast(tf.ones((emb.shape[0], 1)) * decoder_input, dtype=tf.int32)
                nsamples = embedding.shape[0]
                eos_lst = [0] * nsamples
            for i in range(self.seq_len):
                nattn = expected_len if expected_len is not None else i + 1

                enc_input_dummy = self.make_dummy_input(
                    expected_len=expected_len, nattn=nattn, batch_size=emb.shape[0])
                enc_padding_mask, combined_mask, dec_padding_mask = builders.utils.create_masks(
                    enc_input_dummy, output)

                # predictions.shape == (batch_size, seq_len, vocab_size)
                res = self.decode(embedding, output, dec_padding_mask, combined_mask, False)
                # select the last word from the seq_len dimension
                predictions = res['recon'][:, -1:, ...]  # (batch_size, 1, vocab_size)

                if self.dataset.hps['use_continuous_data']:
                    predicted = predictions
                    predicted = tf.concat((predicted[..., :2],
                                           tf.nn.softmax(predicted[..., 2:], axis=-1)), axis=-1)
                    finished_ones = np.sum(tf.argmax(predicted[...,  2:], axis=-1) == 2)
                    output = tf.concat([output, predicted], axis=1)
                    if finished_ones == emb.shape[0]:
                        break
                else:
                    predicted = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
                    output = tf.concat([output, predicted], axis=1)
                    for eos_id in np.where(predicted.numpy().squeeze() == self.dataset.tokenizer.EOS)[0]:
                        eos_lst[eos_id] = 1
                    if sum(eos_lst) == nsamples:
                        break
                # concatentate the predicted_id to the output which is given to the decoder
                # as its input.

            out['recon'] = output
            out['attn_weights'] = res['attn_weights']
        if self.hps['do_classification']:
            pred_labels = self.classify_from_embedding(embedding, False)
            pred_labels = tf.cast(tf.argmax(pred_labels, axis=-1), tf.int32)
            out['class'] = pred_labels
        return out

    def prepare_model_trainer(self):

        # define the input shape signature, which depends on data format
        in_shape = (None, None, 5) if self.dataset.hps['use_continuous_data'] else (None, None)
        dtype = tf.float32 if self.dataset.hps['use_continuous_data'] else tf.int64
        train_step_signature = [
            tf.TensorSpec(shape=in_shape, dtype=dtype),
            tf.TensorSpec(shape=in_shape, dtype=dtype),
            tf.TensorSpec(shape=(None, 1), dtype=tf.int64),
        ]

        # build the trainer and save it as an instance variable
        @tf.function(input_signature=train_step_signature)
        def model_trainer(inp, tar, lab):
            tar_inp = tar[:, :-1, ...]
            tar_real = tar[:, 1:, ...]

            enc_padding_mask, combined_mask, dec_padding_mask = builders.utils.create_masks(inp, tar_inp)
            with tf.GradientTape() as tape:
                res = self.call(inp, tar_inp, True, enc_padding_mask, combined_mask, dec_padding_mask)
                all_losses = []
                if self.hps['do_reconstruction']:
                    recon = self.losses_manager.compute_loss('recon', tar_real, res['recon'])
                    all_losses.append(recon)
                    self.metrics_manager.compute('recon_loss', recon)
                    if not self.dataset.hps['use_continuous_data']:
                        self.metrics_manager.compute('recon_acc', tar_real, res['recon'])
                if self.hps['do_classification']:
                    clas = self.losses_manager.compute_loss('class', lab, res['class'])
                    all_losses.append(clas)
                    self.metrics_manager.compute('class_loss', clas)
                    self.metrics_manager.compute('class_acc', lab, res['class'])
                total_loss = sum(all_losses)
                self.metrics_manager.compute('total_loss', total_loss)
            gradients = tape.gradient(total_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.model_trainer = model_trainer

    def train_on_batch(self, batch):

        data, labels = batch
        inp = tar = data
        self.model_trainer(inp, tar, labels)

        # gather metrics and return
        quick_metrics = self.metrics_manager.get_results_as_dict()
        return quick_metrics

    def prepare_for_start_of_epoch(self):
        pass

    def prepare_for_end_of_epoch(self):
        self.metrics_manager.reset()
