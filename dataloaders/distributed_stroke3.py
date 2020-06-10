import numpy as np
import os
import glob
import utils
import time

from core.data import BaseDataLoader, DatasetSplit


class DistributedStroke3DataLoader(BaseDataLoader):
    name = "stroke3-distributed"

    @classmethod
    def default_hparams(cls):
        hps = utils.hparams.HParams(
            max_seq_len=200,
            shuffle_stroke=False,
            token_type='dictionary',  # grid or dictionary
            use_continuous_data=False,
            use_absolute_strokes=False,
            tokenizer_dict_file='prep_data/sketch_token/token_dict.pkl',
            tokenizer_resolution=100,
            augment_stroke_prob=0.1,
            random_scale_factor=0.1,
        )
        return hps

    def __init__(self, hps, data_directory):

        self.limit = 1000

        if not hps.use_continuous_data and hps.token_type == 'dictionary':
            self.tokenizer = utils.Tokenizer(hps.tokenizer_dict_file,
                                             max_seq_len=0)
        elif not hps.use_continuous_data and hps.token_type == 'grid':
            self.tokenizer = utils.GridTokenizer(resolution=100)

        meta_file = [f for f in glob.glob("{}/*".format(data_directory))
                     if os.path.basename(f).startswith('meta')][0]
        meta_dict = np.load(meta_file, allow_pickle=True)
        self.n_classes = int(meta_dict['n_classes'])
        self.n_samples = int(meta_dict['n_samples_train'])
        self.class_names = meta_dict['class_names']
        self.scale_factor = float(meta_dict['std'])

        super().__init__(hps, data_directory)

    def get_data_splits(self):

        train_f = [f for f in glob.glob("{}/*".format(self.data_directory))
                   if os.path.basename(f).startswith('train')]
        test_f = [f for f in glob.glob("{}/*".format(self.data_directory))
                  if os.path.basename(f).startswith('test')]
        valid_f = [f for f in glob.glob("{}/*".format(self.data_directory))
                   if os.path.basename(f).startswith('valid')]
        return [DatasetSplit('train', train_f),
                DatasetSplit('test', test_f),
                DatasetSplit('valid', valid_f)]

    def get_sample(self, data, idx):
        sample = [np.array(data['x'][idx]),
                  np.expand_dims(data['y'][idx], axis=-1)]
        return np.array(sample)

    def reshuffle_file_indices(self, split_name, filenames):
        if split_name == 'train':
            return np.random.permutation(len(filenames))
        else:
            return list(range(len(filenames)))

    def reshuffle_sample_indices(self, split_name, data):
        if split_name == 'train':
            return np.random.permutation(len(data['x']))
        else:
            return list(range(len(data['x'])))

    def load_next_megabatch(self, split_name, selected_file):

        loaded_dict = np.load(selected_file, allow_pickle=True)
        resulting_data_dict = {'x': loaded_dict['x'],
                               'y': loaded_dict['y']}

        augment = split_name == 'train'  # should do data augmentation
        resulting_data_dict['x'] = self.preprocess(resulting_data_dict['x'],
                                                   augment=augment)
        self.set_future_data_for_split(split_name, resulting_data_dict)

        print("[INFO] Loaded megabatch from {}".format(selected_file))

    def preprocess(self, data, augment=False):
        preprocessed = []
        for sketch in data:
            # removes large gaps from the data
            sketch = np.minimum(sketch, self.limit)
            sketch = np.maximum(sketch, -self.limit)
            sketch = np.array(sketch, dtype=np.float32)

            # augment if required
            sketch = self._augment_sketch(sketch) if augment else sketch

            # get bounds of sketch and use them to normalise
            min_x, max_x, min_y, max_y = utils.sketch.get_bounds(sketch)
            max_dim = max([max_x - min_x, max_y - min_y, 1])
            sketch[:, :2] /= max_dim

            # check for distinct preprocessing options
            if self.hps['shuffle_stroke']:
                lines = utils.tu_sketch_tools.strokes_to_lines(sketch, scale=1.0, start_from_origin=True)
                np.random.shuffle(lines)
                sketch = utils.tu_sketch_tools.lines_to_strokes(lines)
            if self.hps['use_absolute_strokes']:
                sketch = utils.sketch.convert_to_absolute(sketch)
            if not self.hps['use_continuous_data']:
                sketch = self.tokenizer.encode(sketch)

            # slice down overgrown sketches
            if len(sketch) > self.hps['max_seq_len']:
                sketch = sketch[:self.hps['max_seq_len']]

            sketch = self._cap_pad_and_convert_sketch(sketch)

            if not self.hps['use_continuous_data']:
                sketch = np.squeeze(sketch)
            preprocessed.append(sketch)
        return np.array(preprocessed)

    def random_scale(self, data):
        """Augment data by stretching x and y axis randomly [1-e, 1+e]."""
        x_scale_factor = (
            np.random.random() - 0.5) * 2 * self.hps['random_scale_factor'] + 1.0
        y_scale_factor = (
            np.random.random() - 0.5) * 2 * self.hps['random_scale_factor'] + 1.0
        result = np.copy(data)
        result[:, 0] *= x_scale_factor
        result[:, 1] *= y_scale_factor
        return result

    def _cap_pad_and_convert_sketch(self, sketch):
        desired_length = self.hps['max_seq_len']
        skt_len = len(sketch)

        if not self.hps['use_continuous_data']:
            converted_sketch = np.ones((desired_length, 1), dtype=int) * self.tokenizer.PAD
            converted_sketch[:skt_len, 0] = sketch
        else:
            converted_sketch = np.zeros((desired_length, 5), dtype=float)
            converted_sketch[:skt_len, 0:2] = sketch[:, 0:2]
            converted_sketch[:skt_len, 3] = sketch[:, 2]
            converted_sketch[:skt_len, 2] = 1 - sketch[:, 2]
            converted_sketch[skt_len:, 4] = 1
            converted_sketch[-1:, 4] = 1

        return converted_sketch

    def _augment_sketch(self, sketch, set_type='train'):
        if self.hps['augment_stroke_prob'] > 0 and set_type == 'train' and self.hps['use_continuous_data']:
            data_raw = self.random_scale(sketch)
            data = np.copy(data_raw)
            data = utils.sketch.augment_strokes(data, self.hps['augment_stroke_prob'])
            return data
        else:
            return sketch

    def preprocess_extra_sets_from_interp_experiment(self, data):
        preprocessed_sketches = []
        for sketch in data:

            if self.hps['use_absolute_strokes']:
                sketch = utils.sketch.convert_to_absolute(sketch)
            if not self.hps['use_continuous_data']:
                sketch = self.tokenizer.encode(sketch)

            if len(sketch) > self.hps['max_seq_len']:
                sketch = sketch[:self.hps['max_seq_len']]

            sketch = self._cap_pad_and_convert_sketch(sketch)
            preprocessed_sketches.append(sketch)

        return np.array(preprocessed_sketches)

    def get_class_exclusive_random_batch(self, split_name, n, class_list):
        """Return a randomized batch from split, with fixed seed
        and containing a balanced set of samples from each of the selected
        classes (class_list) 
        """

        data = self.get_split_data(split_name)
        x, y = data['x'], data['y']

        np.random.seed(14)
        idx = np.random.permutation(len(x))
        np.random.seed()

        n_per_class = n // len(class_list)
        sel_skts = []
        for chosen_class in class_list:
            n_from_class = 0
            for i in idx:
                if y[i] == chosen_class:
                    sel_skts.append(x[i])
                    n_from_class += 1
                    if n_from_class >= n_per_class:
                        break
        return np.array(sel_skts)
