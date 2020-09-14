import os

import numpy as np

import utils
from core.experiments import Experiment


class ExtractSketchEmbeddings(Experiment):
    name = "extract-embeddings"
    requires_model = True

    @classmethod
    def specific_default_hparams(cls):
        hps = utils.hparams.HParams(
            batch_size=256,
            target_file='embeddings.npz',
            set_type='valid',
            n_samples_to_reconstruct=32
        )
        return hps

    def compute(self, model=None):

        # compute reconstructions on n samples
        x, y = model.dataset.get_n_samples_from(
            'valid', n=self.hps['n_samples_to_reconstruct'], shuffled=True, seeded=True)
        pred_x, pred_y, pred_z = [], [], []
        for i in range(0, len(x), self.hps['batch_size']):
            end_idx = i + self.hps['batch_size'] if i + self.hps['batch_size'] < len(x) else len(x)
            batch_x = x[i:end_idx]

            results = model.predict(batch_x)
            pred_x.append(results['recon'])

        # extract embeddings and classification from all samples on set
        all_x, all_y = model.dataset.get_all_data_from(self.hps['set_type'])
        for i in range(0, len(all_x), self.hps['batch_size']):
            end_idx = i + self.hps['batch_size'] if i + self.hps['batch_size'] < len(all_x) else len(all_x)
            batch_x = all_x[i:end_idx]

            results = model.predict_class(batch_x)
            pred_y.append(results['class'])
            pred_z.append(results['embedding'])
        pred_x = np.concatenate(pred_x, axis=0)
        pred_y = np.concatenate(pred_y, axis=0)
        pred_z = np.concatenate(pred_z, axis=0)

        if model.dataset.hps['use_continuous_data']:
            pred_x = utils.sketch.predictions_to_sketches(pred_x)
            x = utils.sketch.predictions_to_sketches(x)
        else:
            pred_x = np.array(model.dataset.tokenizer.decode_list(pred_x))
            x = np.array(model.dataset.tokenizer.decode_list(x))

        np.savez(self.hps['target_file'],
                 y=all_y,
                 sketches=x,
                 recon_sketches=pred_x,
                 pred_y=pred_y,
                 embeddings=pred_z,
                 )
