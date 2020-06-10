"""
tokenizer.py
Created on Oct 04 2019 15:05
class to encode and decode stroke3 into sketch token
@author: Tu Bui tb0035@surrey.ac.uk
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from .helpers import load_pickle
from .skt_tools import strokes_to_lines, lines_to_strokes


class Tokenizer(object):
    """
    tokenize sketches in stroke3 using clustering
    """

    def __init__(self, dict_path, max_seq_len=0):
        """
        initialize dictionary (a sklearn cluster object)
        :param dict_path: path to pickle file
        :param max_seq_len: 0 if variable length sketch
        """
        self.max_seq_len = max_seq_len
        self.dict = load_pickle(dict_path)
        self.PAD = 0
        self.SEP = self.dict.n_clusters + 1  # whole dictionary needs to be shifted by 1
        self.SOS = self.dict.n_clusters + 2
        self.EOS = self.dict.n_clusters + 3
        self.VOCAB_SIZE = self.dict.n_clusters + 4

    def encode(self, stroke3, seq_len=0):
        """
        encode stroke3 into tokens
        :param stroke3: stroke3 data as numpy array (nx3)
        :param seq_len: if positive, the output is padded with PAD
        :return: sequence of integers as list
        """
        stroke3 += np.zeros((1, 3))
        out = self.dict.predict(stroke3[:, :2])
        # shift by 1 due to PAD token
        out = out + 1
        out = list(out)
        # insert SEP token
        positions = np.where(stroke3[:, 2] == 1)[0]
        offset = 1
        for i in positions:
            out.insert(i + offset, self.SEP)
            offset += 1
        # insert SOS and EOS
        out = [self.SOS] + out + [self.EOS]
        if self.max_seq_len:  # pad
            npad = self.max_seq_len - len(out)
            if npad > 0:
                out += [self.PAD] * npad
            else:
                out = out[:self.max_seq_len]
                out[-2:] = [self.SEP, self.EOS]
        if len(out) < seq_len:
            out += [self.PAD] * (seq_len-len(out))
        return np.array(out)

    def decode(self, seqs):
        if len(seqs) > 0 and isinstance(seqs[0], (list, tuple, np.ndarray)):
            return self.decode_list(seqs)
        else:
            return self.decode_single(seqs)

    def decode_single(self, seq):
        """
        decode a sequence of token id to stroke3
        :param seq: list of integer
        :return: stroke3 array (nx3)
        """
        cluster_ids = []
        pen_states = []
        for i in seq:
            if i not in [self.SOS, self.EOS, self.SEP, self.PAD]:
                cluster_ids.append(i)
                pen_states.append(0)
            elif i == self.SEP and len(pen_states) > 0:
                pen_states[-1] = 1
            elif i == self.EOS:
                break
        if len(cluster_ids) > 0:
            cluster_ids = np.array(cluster_ids)
            cluster_ids = cluster_ids - 1
            dxy = self.dict.cluster_centers_[cluster_ids]
            out = np.c_[dxy, np.array(pen_states)]
            return np.array(out)
        else:
            return np.zeros((1, 3), dtype=np.float32)  #empty sketch

    def decode_list(self, sketches):
        decoded = []
        for s in sketches:
            decoded.append(self.decode_single(np.squeeze(s)))
        return decoded


class GridTokenizer(object):
    """
    tokenize sketches via griding
    Each point will be associated with the nearest grid ID
    """

    def __init__(self, resolution=100, max_seq_len=0):
        """
        :param resolution: grid size will be resolution**2
        """
        self.max_seq_len = max_seq_len
        self.r = int(resolution / 2)  # norm stroke3 has dx,dy \in [-1, 1]
        self.resolution = 2 * self.r  # resolution along each axis
        self.half_pixel = 1 / self.resolution  # radius of a grid cell, used to offset to cell center
        # define 4 special tokens
        self.PAD = 0
        self.SEP = self.resolution**2 + 1
        self.SOS = self.SEP + 1
        self.EOS = self.SEP + 2

        self.VOCAB_SIZE = self.resolution**2 + 4

    def encode(self, stroke3, seq_len=0):
        """
        convert stroke3 to tokens
        :param stroke3: array (N,3); sketch has max size = 1.0
        :param seq_len: if positive, the output is padded with PAD
        :return: list of tokens
        """
        # stroke3 -> stroke3s
        stroke3s = strokes_to_lines(stroke3, 1.0)  # make absolute (x,y)

        # stroke3s -> tokens
        out = []
        for stroke in stroke3s:
            x_t = np.int64((stroke[:, 0] + 1) * self.r)
            x_t[x_t == self.resolution] = self.resolution - 1  # deal with upper bound
            y_t = np.int64((stroke[:, 1] + 1) * self.r)
            y_t[y_t == self.resolution] = self.resolution - 1
            t_id = x_t + y_t * self.resolution
            t_id = list(t_id + 1) + [self.SEP]  # shift by 1 to reserve id 0 for PAD
            out.extend(t_id)
        out = [self.SOS] + out + [self.EOS]
        if self.max_seq_len:  # pad
            npad = self.max_seq_len - len(out)
            if npad > 0:
                out += [self.PAD] * npad
            else:
                out = out[:self.max_seq_len]
                out[-2:] = [self.SEP, self.EOS]
        if len(out) < seq_len:
            out += [self.PAD] * (seq_len-len(out))
        return np.array(out)

    def decode(self, seqs):
        if len(seqs) > 0 and isinstance(seqs[0], (list, tuple, np.ndarray)):
            return self.decode_list(seqs)
        else:
            return self.decode_single(seqs)

    def decode_single(self, tokens):
        """
        convert tokens to stroke3 format
        :param tokens: list of sketch tokens
        :return: stroke3 as (N,3) array
        """
        stroke3s = []
        line = []
        for token in tokens:
            if 0 < token < self.SEP:  # not a special tokens
                y_t = (token - 1) // self.resolution
                x_t = (token - 1) % self.resolution
                y_t = y_t / self.r - 1 + self.half_pixel
                x_t = x_t / self.r - 1 + self.half_pixel
                line.append(np.array([x_t, y_t]))
            elif token == self.SEP and line:
                stroke3s.append(np.array(line))
                line = []
            elif token == self.EOS:
                break
        if line != []:
            stroke3s.append(np.array(line))
        if stroke3s == []:
            stroke3s.append(np.array([[0., 0.]]))
        stroke3 = lines_to_strokes(stroke3s, omit_first_point=False)
        return stroke3

    def decode_list(self, sketches):
        decoded = []
        for s in sketches:
            try:
                decoded.append(self.decode_single(np.squeeze(s)))
            except:
                continue
        return decoded
