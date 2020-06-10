import threading
from abc import ABCMeta, abstractmethod
import numpy as np
import pickle


class QuickMetric(object):
    """Represent a numerical metric that is computed every step of training.
    After each step, BaseModel will call `append_to_history` on each of 
    its quick metrics, keeping a record of all previous states. Using `save`
    and `load` allows one to easily recover plots
    """

    def __init__(self):
        self.last_value = 0.
        self.history = [0.]

    def append_to_history(self, new_value):
        if self.history == [0.]:
            self.history = []
        self.last_value = new_value
        self.history.append(new_value)

    def get_mean_of_latest(self, n=1000):
        if len(self.history) > n:
            return np.mean(self.history[-n:])
        elif len(self.history) == 0:
            return 0
        else:
            return np.mean(self.history)

    def get_std_of_latest(self, n=1000):
        if len(self.history) > n:
            return np.std(self.history[-n:])
        elif len(self.history) == 0:
            return 0
        else:
            return np.std(self.history)

    def save(self, filepath):
        dict_repr = {
            "last_value": self.last_value,
            "history": self.history,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(dict_repr, f)

    def load(self, filepath):
        with open(filepath, "rb") as f:
            dict_repr = pickle.load(f)
            self.last_value = dict_repr["last_value"]
            self.history = dict_repr["history"]


class SlowMetric(object, metaclass=ABCMeta):
    """Abstract base class for metrics that are computed sparingly using
    the current state of the model. Usage is meant to be done by overriding
    the children (HistoryMetric, ProjectionMetric, etc.) and providing
    a `compute` method, as well as specifying a name and required input data.

    See the metrics package for examples.
    """

    def __init__(self, params):
        self.hps = params
        self.thread = threading.Thread()
        self.thread.start()

    def compute_in_parallel(self, input_data):
        self.thread.join()  # wait for previous computation to finish
        self.thread = threading.Thread(target=self.computation_worker,
                                       args=(input_data,))
        self.thread.start()
        # self.thread.join() # debug

    @abstractmethod
    def computation_worker(self, input_data):
        pass

    @abstractmethod
    def compute(self):
        pass

    @abstractmethod
    def get_data_for_plot(self):
        pass

    @abstractmethod
    def is_ready_for_plot(self):
        pass

    @property
    @abstractmethod
    def plot_type(self):
        pass

    @property
    @abstractmethod
    def input_type(self):
        pass

    @property
    @abstractmethod
    def last_value_repr(self):
        return ''

    @abstractmethod
    def save(self, filepath):
        pass

    @abstractmethod
    def load(self, filepath):
        pass


class HistoryMetric(SlowMetric):
    plot_type = 'lines'

    def __init__(self, params):
        super().__init__(params)
        self.history = []
        self.last_value = None

    def computation_worker(self, input_data):
        try:
            result = self.compute(input_data)
        except Exception as e:
            print("Exception while computing metrics: {}".format(repr(e)))
            if self.last_value is None:
                result = 0
            else:
                result = self.last_value
        self.last_value = result
        self.history.append(result)

    def get_data_for_plot(self):
        return self.history

    def is_ready_for_plot(self):
        return bool(self.history)

    @property
    def last_value_repr(self):
        if self.last_value is not None:
            return str(self.last_value)
        else:
            return 'waiting'

    def save(self, filepath):
        if self.last_value is not None:
            dict_repr = {
                "last_value": self.last_value,
                "history": self.history,
            }
            with open(filepath, 'wb') as f:
                pickle.dump(dict_repr, f)

    def load(self, filepath):
        with open(filepath, "rb") as f:
            dict_repr = pickle.load(f)
            self.last_value = dict_repr["last_value"]
            self.history = dict_repr["history"]


class ProjectionMetric(SlowMetric):
    plot_type = 'scatter'

    def __init__(self, params):
        super().__init__(params)
        self.current_projection = None

    def computation_worker(self, input_data):
        try:
            result = self.compute(input_data)
        except Exception as e:
            print("Exception while computing metrics: {}".format(repr(e)))
            if self.current_projection is None:
                result = np.array([[0., 0., 0.]])
            else:
                result = self.current_projection
        self.current_projection = result

    def get_data_for_plot(self):
        return self.current_projection

    def is_ready_for_plot(self):
        return self.current_projection is not None

    @property
    def last_value_repr(self):
        if self.current_projection is not None:
            return 'plotted'
        else:
            return 'waiting'

    def save(self, filepath):
        if self.current_projection is not None:
            dict_repr = {
                "current_projection": self.self.current_projection
            }
            with open(filepath, 'wb') as f:
                pickle.dump(dict_repr, f)

    def load(self, filepath):
        with open(filepath, "rb") as f:
            dict_repr = pickle.load(f)
            self.current_projection = dict_repr["current_projection"]


class ImageMetric(SlowMetric):
    plot_type = 'image'

    def __init__(self, params):
        super().__init__(params)
        self.current_image = None

    def computation_worker(self, input_data):
        try:
            result = self.compute(input_data)
        except Exception as e:
            print("Exception while computing metrics: {}".format(repr(e)))
        self.current_image = result

    def get_data_for_plot(self):
        return self.current_image

    def is_ready_for_plot(self):
        return self.current_image is not None

    @property
    def last_value_repr(self):
        if self.is_ready_for_plot:
            return 'image-grid'
        else:
            return 'waiting'

    def save(self, filepath):
        if self.is_ready_for_plot:
            dict_repr = {
                "current_image": self.self.current_image
            }
            with open(filepath, 'wb') as f:
                pickle.dump(dict_repr, f)

    def load(self, filepath):
        with open(filepath, "rb") as f:
            dict_repr = pickle.load(f)
            self.current_image = dict_repr["current_image"]


class HistogramMetric(SlowMetric):
    plot_type = 'hist'

    def __init__(self, params):
        super().__init__(params)
        self.current_hist = None

    def computation_worker(self, input_data):
        try:
            result = self.compute(input_data)
        except Exception as e:
            print("Exception while computing metrics: {}".format(repr(e)))
            if self.current_hist is None:
                result = np.array([0.])
            else:
                result = self.current_hist
        self.current_hist = result

    def get_data_for_plot(self):
        return self.current_hist

    def is_ready_for_plot(self):
        return self.current_hist is not None

    @property
    def last_value_repr(self):
        if self.current_hist is not None:
            return 'plotted'
        else:
            return 'waiting'

    def save(self, filepath):
        if self.is_ready_for_plot:
            dict_repr = {
                "current_hist": self.self.current_hist
            }
            with open(filepath, 'wb') as f:
                pickle.dump(dict_repr, f)

    def load(self, filepath):
        with open(filepath, "rb") as f:
            dict_repr = pickle.load(f)
            self.current_hist = dict_repr["current_hist"]
