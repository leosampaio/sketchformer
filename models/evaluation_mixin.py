import os

import numpy as np

import utils


class TransformerMetricsMixin(object):

    def compute_predictions_on_validation_set(self):
        return self.compute_predictions_on_set('valid')

    def compute_predictions_on_test_set(self):
        return self.compute_predictions_on_set('test')

    def compute_predictions_on_set(self, set_type):
        # do reconstruction with smaller batch
        x, y = self.dataset.get_n_samples_from(
            'valid', n=32, shuffled=True, seeded=True)
        pred_x, pred_y, pred_z = [], [], []
        for i in range(0, len(x), self.hps['batch_size']):
            end_idx = i + self.hps['batch_size'] if i + self.hps['batch_size'] < len(x) else len(x)
            batch_x = x[i:end_idx]

            results = self.predict(batch_x)
            pred_x.append(results['recon'])

        all_x, all_y = self.dataset.get_all_data_from(set_type)
        for i in range(0, len(all_x), self.hps['batch_size']):
            end_idx = i + self.hps['batch_size'] if i + self.hps['batch_size'] < len(all_x) else len(all_x)
            batch_x = all_x[i:end_idx]

            results = self.predict_class(batch_x)
            pred_y.append(results['class'])
            pred_z.append(results['embedding'])

        pred_x = np.concatenate(pred_x, axis=0)
        pred_y = np.concatenate(pred_y, axis=0)
        pred_z = np.concatenate(pred_z, axis=0)

        if self.dataset.hps['use_continuous_data']:
            pred_x = utils.sketch.predictions_to_sketches(pred_x)
            x = utils.sketch.predictions_to_sketches(x)

        time_id = utils.helpers.get_time_id_str()
        plot_filepath = os.path.join(self.plots_out_dir,
                                     "{}_{}.svg".format(time_id, '{}'))
        tmp_filepath = os.path.join(self.tmp_out_dir, "converted_{}.png")
        if not self.dataset.hps['use_continuous_data']:
            tokenizer = self.dataset.tokenizer
        else:
            tokenizer = None
        return x, all_y, pred_x, pred_y, pred_z, tokenizer, plot_filepath, tmp_filepath, self.dataset.hps['use_continuous_data']
