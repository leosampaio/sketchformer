import os
import pickle
import time
import pprint
import socket
from abc import ABCMeta, abstractmethod

import tensorflow as tf

import utils
import metrics
from core.metrics import QuickMetric
from core.notifyier import Notifyier


class BaseModel(tf.keras.Model, metaclass=ABCMeta):

    @classmethod
    def base_default_hparams(cls):
        base_hparams = utils.hparams.HParams(
            batch_size=128,
            num_epochs=10,
            save_every=1.,  # checkpoint every n epochs
            safety_save=.5,  # safety checkpoint is more frequent but is not kept
            autograph=True,  # use tf autograph or not (useful for debugging)
            log_every=100,  # print a log every n steps
            notify_every=1000,  # send a notification and compute slow metrics every n steps
            slack_config='token.secret',  # file with slack setup (token and channel)
            goal='No description',  # describe what this is supposed to achieve
        )
        return base_hparams

    def __init__(self, hps, dataset, outdir, experiment_id):
        """Prepares model for building and training

        :param hps: HParam that should be filled with a call to default_hparams
            and updated with desired changes
        :param dataset: Data loader, child of core.data.BaseDataLoader
        :param outdir: Main output directory, folders for checkpoints and 
            partial results will be created here
        :param experiment_id: String that will be used create the outdir
            and send notifications, should identify your current experiment
        """

        super().__init__()
        self.hps = hps if isinstance(hps, dict) else dict(hps.values())
        self.dataset = dataset
        self.host = socket.gethostname()
        self.experiment_id = experiment_id
        self.batches_per_epoch = self.dataset.n_samples // self.hps['batch_size']

        # check if children are not messing up
        if not hasattr(self, 'name'):
            raise Exception('You must give your model a reference name')
        if not hasattr(self, 'quick_metrics'):
            raise Exception("You must define your model's expected quick metric "
                            "names in a quick_metrics class variable")
        if not hasattr(self, 'slow_metrics'):
            raise Exception("You must define your model's expected slow metric "
                            "names in a slow_metrics class variable")

        # prepare both metric dictionaries
        self.quick_metrics = {q: QuickMetric() for q in self.quick_metrics}
        self.slow_metrics = {m: metrics.build_metric_by_name(m, self.hps)
                             for m in self.slow_metrics}

        # create all the output directories if they do not exist
        self.out_dir = os.path.join(outdir, self.identifier)
        self.legacy_out_dir = os.path.join(outdir, self.experiment_id)
        if not os.path.isdir(self.out_dir):
            os.mkdir(self.out_dir)
        self.plots_out_dir = os.path.join(self.out_dir, 'plots')
        if not os.path.isdir(self.plots_out_dir):
            os.mkdir(self.plots_out_dir)
        self.wgt_out_dir = os.path.join(self.out_dir, 'weights')
        if not os.path.isdir(self.wgt_out_dir):
            os.mkdir(self.wgt_out_dir)
        self.tmp_out_dir = os.path.join(self.out_dir, 'tmp')
        if not os.path.isdir(self.tmp_out_dir):
            os.mkdir(self.tmp_out_dir)
        self.config_filepath = self.get_config_filepath(outdir, self.experiment_id)

        # setup notifyier. it sends partial results inclusing slow metrics
        self.notifyier = Notifyier(self.hps['slack_config'])

        # turning off the autograph helps with debugging
        if not self.hps['autograph']:
            tf.config.experimental_run_functions_eagerly(True)

        # prepare step counting; this variable is restored with checkpoints
        self.current_step = tf.Variable(0, name='global_step', trainable=False)

        # finally, build model and checkpoint manager
        self.build_model()
        self.prepare_checkpointing()

    def update_json(self):
        """Save current HParams setup to a json file in self.outdir
        """

        with open(os.path.join(self.out_dir, 'config.json'), 'w') as f:
            json = self.hps.to_json(indent=2, sort_keys=True)
            f.write(json)

    @property
    def identifier(self):
        return "{}-{}".format(self.name, self.experiment_id)

    @classmethod
    def default_hparams(cls):
        """Returns the default hparams for this model 
        """
        return utils.hparams.combine_hparams_into_one(
            cls.specific_default_hparams(),
            cls.base_default_hparams())

    @classmethod
    def get_config_filepath(cls, output_dir, exp_id):
        return os.path.join(output_dir, "{}-{}".format(cls.name, exp_id),
                            'config.json')

    @classmethod
    def parse_hparams(cls, base, specific):
        """Overrides both default sets of parameters (base and specific) and
        combines them into a single HParams objects, which is returned to 
        the caller.

        It is done this way so that the caller can save those parameters
        however they want to
        """
        hps = cls.default_hparams()
        if base is not None:
            hps = hps.parse(base)
        if specific is not None:
            hps = hps.parse(specific)
        return hps

    @classmethod
    @abstractmethod
    def specific_default_hparams(cls):
        """Children should provide their own list of hparams; those will be
        combined with with the base hparams on base_default_hparams and then
        returned by the default_hparams property getter
        """
        pass

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def train_on_batch(self, batch):
        pass

    @abstractmethod
    def prepare_for_start_of_epoch(self):
        pass

    @abstractmethod
    def prepare_for_end_of_epoch(self):
        pass

    def train(self):
        '''Main learning loop
        Take one batch from the dataset loader, pass it through
        the train_on_batch method defined on the child, get the quick metrics
        for history. Keeps track of steps, epochs and notifications
        '''

        # send model description for later reference
        model_descriptor = pprint.pformat(self.hps)
        message = "*Training started on {}*\n*Goal:* {}\nParams:\n{}".format(self.host,
                                                                             self.hps['goal'],
                                                                             model_descriptor)
        self.notifyier.notify_with_message(message, self.identifier)

        # start training loop
        total_steps = self.batches_per_epoch * self.hps['num_epochs']
        self.epoch = self.current_step // self.batches_per_epoch

        data_iterator = self.dataset.batch_iterator(
            split_name='train', batch_size=self.hps['batch_size'], stop_at_end_of_split=False)
        for _ in range(total_steps - self.current_step):
            self.current_step.assign_add(1)

            # train on current batch
            quick_metrics = self.train_on_batch(next(data_iterator))

            # check for required reports and checkpoints
            self.update_quick_metrics_history(quick_metrics)
            self.status_report()
            self.save_checkpoint_if_its_time()

            # end of epoch
            if self.current_step // self.batches_per_epoch > self.epoch:
                self.epoch = self.current_step // self.batches_per_epoch
                self.status_report(end_of_epoch=True)

    def update_quick_metrics_history(self, new_metrics):
        for qm, q_metric in new_metrics.items():
            self.quick_metrics[qm].append_to_history(q_metric)

    def status_report(self, end_of_epoch=False):

        # prepare start of log string
        cur_iter = self.current_step.numpy() % self.batches_per_epoch
        batch_info_str = "Batch {}/{}".format(cur_iter, self.batches_per_epoch)
        log_string = "Epoch {} {}" .format(self.epoch, batch_info_str)

        # include quick metrics reporting
        for k, l in self.quick_metrics.items():
            log_string = "{}|{}={:4.4f}".format(log_string, k, l.last_value)

        # check if required and prepare, plot and send slow metrics
        if (cur_iter != 0 and cur_iter % self.hps['notify_every'] == 0) or end_of_epoch:
            log_string = self.prepare_plot_send_slow_metrics(log_string)

        # print out log string if required
        if ((cur_iter % self.hps['log_every'] == 0)
                or (cur_iter % self.hps['notify_every'] == 0)
                or end_of_epoch):
            print(log_string)
        return log_string

    def prepare_plot_send_slow_metrics(self, msg):
        self.compute_all_metrics()
        plot_file = self.plot_all_metrics()
        for sm, metric in self.slow_metrics.items():
            msg = "{}, {}={}".format(msg, sm, metric.last_value_repr)
        self.notifyier.notify_with_image(plot_file,
                                         self.identifier, msg)
        return msg

    def plot_all_metrics(self):
        plot_manager = utils.plots.PlotManager()
        for m, metric in self.quick_metrics.items():
            plot_manager.add_metric(
                name=m, data=metric.history, data_type='lines', skipped_iterations=1)
        for m, metric in self.slow_metrics.items():
            if metric.is_ready_for_plot():
                plot_manager.add_metric(
                    name=m, data=metric.get_data_for_plot(),
                    data_type=metric.plot_type, skipped_iterations=self.hps['notify_every'])
        tmp_file = os.path.join(self.plots_out_dir,
                                "s{}_plots.png".format(utils.helpers.get_time_id_str()))
        plot_manager.plot(tmp_file, figsize=8, wspace=0.15)
        return tmp_file

    def compute_all_metrics(self):
        for m, metric in self.slow_metrics.items():
            input_data = self.gather_data_for_metric(metric.input_type)
            metric.compute_in_parallel(input_data)

    def compute_metrics_from(self, chosen_metrics):
        for m, metric in chosen_metrics.items():
            input_data = self.gather_data_for_metric(metric.input_type)
            metric.computation_worker(input_data)

    def plot_and_send_notification_for(self, metrics_list):
        plot_manager = utils.plots.PlotManager()
        for m, metric in metrics_list.items():
            plot_manager.add_metric(
                name=m, data=metric.get_data_for_plot(),
                data_type=metric.plot_type, skipped_iterations=self.hps['notify_every'])
        tmp_file = os.path.join(self.plots_out_dir,
                                "evaluation_plots.png")
        plot_manager.plot(tmp_file, figsize=8, wspace=0.15)
        self.notifyier.notify_with_image(tmp_file,
                                         self.identifier)

    def clean_up_tmp_dir(self):
        for the_file in os.listdir(self.tmp_out_dir):
            file_path = os.path.join(self.tmp_out_dir, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)

    def gather_data_for_metric(self, data_type):
        data = self.load_precomputed_features_if_they_exist(data_type)
        if not data:
            try:
                gather_func = getattr(self, "compute_{}".format(data_type))
            except AttributeError:
                raise AttributeError("One of your metrics requires a "
                                     "compute_{} method".format(data_type))
            data = gather_func()
            self.save_precomputed_features(data_type, data)
            data = self.load_precomputed_features_if_they_exist(data_type)
        return data

    def load_precomputed_features_if_they_exist(self, feature_type):
        time_id = utils.helpers.get_time_id_str()
        filename = os.path.join(
            self.tmp_out_dir,
            "precomputed_{}_{}.pickle".format(feature_type, time_id))

        if os.path.exists(filename):
            try:
                with open(filename, 'rb') as f:
                    data = pickle.load(f)
                    return data
            except EOFError as _:
                return False
        else:
            return False

    def save_precomputed_features(self, feature_type, data):
        utils.helpers.remove_latest_similar_file_if_it_exists(os.path.join(
            self.tmp_out_dir, "precomputed_{}*".format(feature_type)))
        time_id = utils.helpers.get_time_id_str()
        start = time.time()
        filename = os.path.join(
            self.tmp_out_dir,
            "precomputed_{}_{}.pickle".format(feature_type, time_id))
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print("[Precalc] Saving {} took {}s".format(feature_type, time.time() - start))

    def prepare_checkpointing(self):
        self.ckpt = tf.train.Checkpoint(transformer=self,
                                        optimizer=self.optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(
            self.ckpt, self.wgt_out_dir, max_to_keep=2)
        # if there are no current checkpoints, maybe it has legacy ones
        if not self.ckpt_manager.latest_checkpoint:
            legacy_manager = tf.train.CheckpointManager(
                self.ckpt, self.legacy_out_dir, max_to_keep=2)
            if legacy_manager.latest_checkpoint:
                self.ckpt_manager = legacy_manager

    def restore_checkpoint_if_exists(self, checkpoint):
        if checkpoint is None:
            return
        if checkpoint == 'latest':
            latest_checkpoint = tf.train.latest_checkpoint(self.wgt_out_dir)
            if latest_checkpoint is not None:
                self.ckpt.restore(latest_checkpoint)
                print("[Checkpoint] Restored, step #{}".format(self.current_step.numpy()))
            else:
                print("[Checkpoint] Not found")
        else:
            self.ckpt.restore(checkpoint)

    def save_checkpoint_if_its_time(self):
        cur_steps = self.current_step.numpy()
        safety_save = int(self.hps['safety_save'] * self.batches_per_epoch)
        save_every = int(self.hps['save_every'] * self.batches_per_epoch)
        if (int(cur_steps) + 1) % safety_save == 0:
            ckpt_save_path = self.ckpt_manager.save()
            print('Saving safety checkpoint for step {} at {}'.format(
                self.current_step + 1, ckpt_save_path))
        if (int(cur_steps) + 1) % save_every == 0:
            filename = "{}/step{}".format(self.wgt_out_dir, cur_steps)
            self.ckpt.save(file_prefix=filename)
            print('Saving fixed checkpoint for step {} at {}'.format(
                self.current_step + 1, filename))
