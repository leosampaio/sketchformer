import os

import numpy as np

import utils
from core.experiments import Experiment


class InterpolationsForMturk(Experiment):
    name = "interpolations-for-mturk"
    requires_model = True

    @classmethod
    def specific_default_hparams(cls):
        hps = utils.hparams.HParams(
            source_emb='/stornext/CVPNOBACKUP/scratch_4weeks/m13395/experiments_output/cvpr_sketch_list/10/source.npz',
            intra_emb='/stornext/CVPNOBACKUP/scratch_4weeks/m13395/experiments_output/cvpr_sketch_list/10/des_intra.npz',
            inter_emb='/stornext/CVPNOBACKUP/scratch_4weeks/m13395/experiments_output/cvpr_sketch_list/10/des_intra.npz',
            batch_size=256
        )
        return hps

    def compute(self, model=None):

        src = np.load(self.hps['source_emb'], allow_pickle=True)
        intra = np.load(self.hps['intra_emb'], allow_pickle=True)
        inter = np.load(self.hps['inter_emb'], allow_pickle=True)
        print("[Interpolation] Loaded {}".format(self.hps['source_emb']))

        x_data = {'src': src['data'], 'intra': intra['data'], 'inter': inter['data']}
        y_data = {'src': src['cat'], 'intra': intra['cat'], 'inter': inter['cat']}
        id_data = {'src': src['ids'], 'intra': intra['ids'], 'inter': inter['ids']}
        z_data = {'src': [], 'intra': [], 'inter': []}
        z_interp = {'inter': [], 'intra': []}
        x_recons_interp = {'inter': [], 'intra': []}
        filename_interp = {'inter': [], 'intra': []}
        x_data_recons = {'data': [], 'cat': src['cat'], 'ids': src['ids']}

        print("[Interpolation] Preprocessing...")

        # preprocess
        for set_type in x_data.keys():
            x_data[set_type] = self.dataset.preprocess_extra_sets_from_interp_experiment(x_data[set_type])
            x_data[set_type] = np.squeeze(x_data[set_type])

        # prepare saving folders
        interp_out_dir = os.path.join(self.out_dir, 'interpolations')
        if not os.path.isdir(interp_out_dir):
            os.mkdir(interp_out_dir)
        interp_out_dirs = {}
        interp_out_dirs['inter'] = os.path.join(interp_out_dir, 'inter')
        if not os.path.isdir(interp_out_dirs['inter']):
            os.mkdir(interp_out_dirs['inter'])
        interp_out_dirs['intra'] = os.path.join(interp_out_dir, 'intra')
        if not os.path.isdir(interp_out_dirs['intra']):
            os.mkdir(interp_out_dirs['intra'])

        reconstruction_out_dir = os.path.join(self.out_dir, 'reconstructions')
        if not os.path.isdir(reconstruction_out_dir):
            os.mkdir(reconstruction_out_dir)

        # gather all z values
        print("[Interpolation] Gathering embedding all sets...")
        for set_type in x_data.keys():

            for i in range(0, len(x_data[set_type]), self.hps['batch_size']):
                end_idx = i + self.hps['batch_size'] if i + self.hps['batch_size'] < len(x_data[set_type]) else len(x_data[set_type])
                batch_x = x_data[set_type][i:end_idx]
                results = model.predict_class(batch_x)
                z_data[set_type].append(results['embedding'])
            z_data[set_type] = np.concatenate(z_data[set_type], axis=0)

        # reconstruction on the srcs
        print("[Interpolation] Reconstructing source samples")
        for i in range(0, len(z_data['src']), self.hps['batch_size']):
            end_idx = i + self.hps['batch_size'] if i + self.hps['batch_size'] < len(z_data['src']) else len(z_data['src'])
            batch_z = z_data['src'][i:end_idx]
            results = model.predict_from_embedding(batch_z, expected_len=None)
            x_data_recons['data'].append(results['recon'].numpy())
        reconstructions_list = []
        for batch in x_data_recons['data']:
            for sample in batch:
                reconstructions_list.append(sample)
        x_data_recons['data'] = reconstructions_list

        if not model.hps['use_continuous_data']:
            x_data_recons['data'] = model.dataset.tokenizer.decode_list(x_data_recons['data'])
        else:
            x_data_recons['data'] = utils.sketch.predictions_to_sketches(x_data_recons['data'])

        print("[Interpolation] Saving reconstructed source samples")
        filepath = os.path.join(reconstruction_out_dir, 'reconstructed_source.npz')
        np.savez(filepath,
                 recon=x_data_recons['data'],
                 cat=x_data_recons['cat'],
                 ids=x_data_recons['ids'])

        # create all possible interpolations

        print("[Interpolation] Creating interpolations...")
        n_inter = 10
        for z_src, z_intra, z_inter in zip(z_data['src'], z_data['intra'], z_data['inter']):

            for t in np.linspace(0, 1, n_inter):
                z_interp['inter'].append(
                    utils.tu_sketch_tools.slerp(z_src, z_inter, t))
                z_interp['intra'].append(
                    utils.tu_sketch_tools.slerp(z_src, z_intra, t))

        # get all reconstructions from interpolations
        for set_type in z_interp.keys():
            z_interp[set_type] = np.array(z_interp[set_type])

            for i in range(0, len(z_interp[set_type]), self.hps['batch_size']):
                print("[Interpolation] Generating sketches for {} set... {}/{}".format(set_type, i, len(z_interp[set_type])), end="\r")
                end_idx = i + self.hps['batch_size'] if i + self.hps['batch_size'] < len(z_interp[set_type]) else len(z_interp[set_type])
                batch_z = z_interp[set_type][i:end_idx]
                results = model.predict_from_embedding(batch_z, expected_len=None)
                x_recons_interp[set_type].append(results['recon'].numpy())
            reconstructions_list = []
            for batch in x_recons_interp[set_type]:
                for sample in batch:
                    reconstructions_list.append(sample)

            if not model.hps['use_continuous_data']:
                x_recons_interp[set_type] = model.dataset.tokenizer.decode_list(reconstructions_list)
            else:
                x_recons_interp[set_type] = utils.sketch.predictions_to_sketches(reconstructions_list)

            # go over the resulting reconstructions and save them
            print("\n[Interpolation] Saving generated sketches for {} set".format(set_type))
            for i in range(len(id_data['src'])):
                filename = "{:03d}_slerp_{}_{}_{}_{}.svg".format(
                    i, y_data['src'][i], y_data[set_type][i],
                    id_data['src'][i], id_data[set_type][i])
                filepath = os.path.join(interp_out_dirs[set_type], filename)
                filename_interp[set_type].append(filepath)

                start_id, end_id = i * n_inter, (i * n_inter) + n_inter
                interp_results = x_recons_interp[set_type][start_id:end_id]
                interp_results = np.nan_to_num(interp_results)
                for i, sketch in enumerate(interp_results):
                    interp_results[i] = np.nan_to_num(sketch)
                sketch_grid = [(interp_results[t], (0, t)) for t in range(len(interp_results))]
                sketch_out = utils.tu_sketch_tools.make_grid_svg(
                    sketch_grid, 1, len(interp_results) * 0.2)
                utils.tu_sketch_tools.draw_strokes3(sketch_out, 0.01, filepath)
