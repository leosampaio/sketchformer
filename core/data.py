import threading
from functools import wraps
from abc import ABCMeta, abstractmethod

import numpy as np


class DatasetSplit(object):
    """Represent a dataset split such as training or validation. Is meant to
    be used as an organized dictionary to help BaseDataLoader
    """

    def __init__(self, name, filepaths):
        self.name, self.filepaths = name, filepaths
        self.n_files = len(self.filepaths)
        self.file_shuffle = list(range(self.n_files))
        self.file_cursor = 0

        self.next, self.current = None, None
        self.current_len, self.current_index_shuffle, self.cursor = 0, [0], 0
        self.did_load_first_megabatch = False
        self.thread = None


class BaseDataLoader(object, metaclass=ABCMeta):
    """All data loaders should inherit from this one. This provides common
    functionality for parallel data loading. The children must implement
    the interface of abstractmethods.

    When inheriting, remember to give your child a name
    """

    def __init__(self, hps, data_directory):

        # check if children are not messing up
        if not hasattr(self, "name"):
            raise Exception("You must give your data loader a reference name")

        self.hps = hps if isinstance(hps, dict) else dict(hps.values())
        self.data_directory = data_directory
        self.splits = self.get_data_splits()
        self.splits = {split.name: split for split in self.splits}

        for s, split in self.splits.items():
            split.file_cursor = 0
            split.file_shuffle = self.reshuffle_file_indices(s, split.filepaths)
            selected_file = split.filepaths[split.file_shuffle[split.file_cursor]]
            split.thread = threading.Thread(
                target=self.load_next_megabatch, args=(s, selected_file))
            split.thread.start()
            # debug
            # split.thread.join()

    @classmethod
    def parse_hparams(cls, params):
        """Overrides the default sets of parameters.

        It is done this way so that the caller can save those parameters
        however they want to
        """
        hps = cls.default_hparams()
        if params is not None:
            hps = hps.parse(params)
        return hps

    def check_if_split_is_ready(function):
        """Decorator to check if the first megabatch of data was loaded 
        from the chosen split. This is useful for every function that uses
        the splits directly
        """
        @wraps(function)
        def wrapper(inst, split_name, *args, **kwargs):
            if not inst.splits[split_name].did_load_first_megabatch:
                inst.splits[split_name].did_load_first_megabatch = True
                inst._swap_used_set_for_preloaded_one(split_name)
            return function(inst, split_name, *args, **kwargs)
        return wrapper

    def _swap_used_set_for_preloaded_one(self, split_name):
        split = self.splits[split_name]

        # swap the megabatches
        split.thread.join()
        split.current = split.next

        # shuffle indices (or not, it is the child's choice)
        split.current_index_shuffle = self.reshuffle_sample_indices(split_name, split.current)
        split.current_len = len(split.current_index_shuffle)

        # advance file cursor and check if we finished the split
        split.file_cursor += 1
        did_run_through_all_data = split.file_cursor == split.n_files
        if did_run_through_all_data:
            split.file_cursor = 0
            split.file_shuffle = self.reshuffle_file_indices(split_name, split.filepaths)

        selected_file = split.filepaths[split.file_shuffle[split.file_cursor]]

        # only load next megabatch if there is more than one to load
        if split.n_files > 1:
            print("[INFO] Swapped data megabatches for split {}".format(split_name))
            split.thread = threading.Thread(
                target=self.load_next_megabatch, args=(split_name, selected_file))
            split.thread.start()
            self.did_swap_megabatches(split)
        return did_run_through_all_data

    def set_future_data_for_split(self, split_name, data):
        """Set next data megabatch for given split. This method must be used
        by the child in load_next_megabatch to guarantee proper functioning
        of the parallel loading scheme
        """
        self.splits[split_name].next = data

    def sample_iterator(self, split_name, batch_size):
        """Iterate through the current index shuffle of the selected split. 
        This index shuffle is a list of indexes that reference the current
        loaded file for the split. 

        This iterator can stop for either returning a full batch of indices,
        or because the split itself was finished (all its files were read
        to completion). The reason is indicated by iterator_status, returned
        on every iteration
        """
        iterator_status = 'running'
        counter = 0
        while iterator_status == 'running':

            # check for end_of_batch, notice that the end_of_split status
            # takes priority as it is set later
            counter += 1
            iterator_status = 'end_of_batch' if counter >= batch_size else 'running'

            # swap files if we reached the end of the current one
            if self.splits[split_name].cursor >= self.splits[split_name].current_len:
                end_of_split = self._swap_used_set_for_preloaded_one(split_name)
                iterator_status = 'end_of_split' if end_of_split else iterator_status
                self.splits[split_name].cursor = 0

            # get the current place in the split (the cursor)
            cursor = self.splits[split_name].current_index_shuffle[self.splits[split_name].cursor]
            self.splits[split_name].cursor += 1

            yield cursor, iterator_status

    @check_if_split_is_ready
    def batch_iterator(self, split_name, batch_size, stop_at_end_of_split):
        """Iterate through batches of samples in selected split (split_name).
        This will use the get_sample function on the child to obtain each
        sample.

        :param split_name: The selected split (one of those returned by
            get_data_splits)
        :param batch_size: Each batch will have this number of samples or
            less (if we reach end of split)
        :param stop_at_end_of_split: Indicates wether the iterator stops at
            end of split (e.g. for eval on test sets) or keeps running forever
            (e.g. for training)
        :return: a python generator that yields (batch_size, *sample) arrays
        """
        iterator_status = 'running'
        while not stop_at_end_of_split or not iterator_status == 'end_of_split':
            samples = []
            for idx, iterator_status in self.sample_iterator(split_name, batch_size):
                sample = self.get_sample(self.splits[split_name].current, idx)
                for i, element in enumerate(sample):
                    if len(samples) == i:
                        samples.append([element])
                    else:
                        samples[i].append(element)
            samples = [np.array(element) for element in samples]
            yield self.preprocess_batch(samples)

    @check_if_split_is_ready
    def get_n_samples_from(self, split_name, n, shuffled=False, seeded=None, preprocess=False):
        """Get n samples from current loaded file of selected split

        :param split_name: The selected split (one of those returned by 
            get_data_splits)
        :param n: number of samples the user wants returned
        :param suffled: to shuffle or not to shuffle indices
        :param seeded: if true, should return the same n samples every call,
            with the disclaimer that for a split that is written over many
            files, this will only guarantee the same samples for the same
            loaded file
        :return: a (n, *sample) array
        """
        if seeded is not None:
            np.random.seed(int(seeded))
        if shuffled:
            indices = np.random.permutation(self.splits[split_name].current_len)[:n]
        else:
            indices = list(range(n))
        if seeded is not None:
            np.random.seed()

        samples = []
        for idx in indices:
            sample = self.get_sample(self.splits[split_name].current, idx)
            for i, element in enumerate(sample):
                if len(samples) == i:
                    samples.append([element])
                else:
                    samples[i].append(element)
        samples = [np.array(element) for element in samples]
        if preprocess:
            return self.preprocess_batch(samples)
        else:
            return samples

    @check_if_split_is_ready
    def get_n_samples_batch_iterator_from(self, split_name, n, batch_size, shuffled=False, seeded=None, preprocess=False):
        """Get n samples from current loaded file of selected split

        :param split_name: The selected split (one of those returned by 
            get_data_splits)
        :param n: number of samples the user wants returned
        :param suffled: to shuffle or not to shuffle indices
        :param seeded: if true, should return the same n samples every call,
            with the disclaimer that for a split that is written over many
            files, this will only guarantee the same samples for the same
            loaded file
        :return: a (n, *sample) array
        """
        if seeded is not None:
            np.random.seed(int(seeded))
        if shuffled:
            indices = np.random.permutation(self.splits[split_name].current_len)[:n]
        else:
            indices = list(range(n))
        if seeded is not None:
            np.random.seed()

        for i in range(0, n, batch_size):
            end_idx = i + batch_size if i + batch_size < n else n
            samples = []
            for idx in indices[i:end_idx]:
                sample = self.get_sample(self.splits[split_name].current, idx)
                for i, element in enumerate(sample):
                    if len(samples) == i:
                        samples.append([element])
                    else:
                        samples[i].append(element)
            samples = [np.array(element) for element in samples]
            yield self.preprocess_batch(samples)

    @check_if_split_is_ready
    def get_all_data_from(self, split_name):
        """Return the split's (split_name) data from current loaded file.
        This is not the same as returning the entire set, as it regards only
        the current loaded file.
        """
        return self.preprocess_batch(self.get_n_samples_from(
            split_name, n=self.splits[split_name].current_len))

    def preprocess_batch(self, samples):
        """Hook for child loaders to add some preprocessing to batches
        """
        return samples

    @classmethod
    @abstractmethod
    def default_hparams(cls):
        """Children should provide their own list of hparams. These should 
        regard preprocessing styles and data formatting in general
        """
        pass

    @abstractmethod
    def get_data_splits(self):
        """Return a list of DatasetSplit objects, containing the name and
        the list of file for each split of the dataset
        """
        pass

    @abstractmethod
    def get_sample(self, data, idx):
        """Return a single sample, using idx as a reference. This should be
        a complete sample, containing all data (e.g. labels and images)
        """
        pass

    @abstractmethod
    def reshuffle_file_indices(self, split_name, filenames):
        """Return a shuffled list of indices referencing the files on
        the filenames list. Use the split_name to control how to shuffle
        (e.g. train splits are shuffled and test splits are not)
        """
        pass

    @abstractmethod
    def reshuffle_sample_indices(self, split_name, data):
        """Return a shuffled list of indices referencing the data on
        the data parameter. Use the split_name to control how to shuffle
        (e.g. train splits are shuffled and test splits are not)
        """
        pass

    @abstractmethod
    def load_next_megabatch(self, split_name, selected_file):
        """Load the selected file, do any necessary preprocessing on the data
        and finally call set_future_data_for_split with the resulting data.
        This will guarantee that parallel loading keeps loading different files
        """
        pass

    def did_swap_megabatches(self, split):
        """Maybe the child wants to do something after a swap (?)
        """
        pass
