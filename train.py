#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_clas_transformer.py
Created on Oct 08 2019 16:08

@author: Tu Bui tb0035@surrey.ac.uk
"""

import argparse
import pprint

import utils
import models
import dataloaders


def main():

    parser = argparse.ArgumentParser(
        description='Train modified transformer with sketch data')
    parser.add_argument("model_name", default=None,
                        help="Model that we are going to train")

    parser.add_argument("--id", default="0", help="experiment signature")
    parser.add_argument("--dataset", default=None, help="Input data folder")
    parser.add_argument("-o", "--output-dir", default="", help="Output directory")

    parser.add_argument("-p", "--hparams", default=None,
                        help="Parameters that are specific to one model. They "
                        "can regard hyperparameters such as number of layers"
                        "or specifics of training such as an optmiser choice")
    parser.add_argument("--base-hparams", default=None,
                        help="Model parameters that concern all models. "
                        "Those are related to logging, checkpointing, "
                        "notifications and loops")
    parser.add_argument("--data-hparams", default=None,
                        help="Dataset-related parameters. Regards data"
                        "formats and preprocessing parameters")

    parser.add_argument("-g", "--gpu", default=0, type=int, nargs='+', help="GPU ID to run on", )
    parser.add_argument("-r", "--resume", default=None, help="One of 'latest' or a checkpoint name")
    parser.add_argument("--data-loader", default='stroke3-distributed',
                        help="Data loader that will provide data for model")
    parser.add_argument("--help-hps", action="store_true",
                        help="Prints out each hparams default values")
    args = parser.parse_args()

    # get our model and data loader classes
    Model = models.get_model_by_name(args.model_name)
    DataLoader = dataloaders.get_dataloader_by_name(args.data_loader)

    # check for desperate help calls from the unending darkness
    if args.help_hps:
        base_help = pprint.pformat(Model.base_default_hparams().values())
        specific_help = pprint.pformat(Model.specific_default_hparams().values())
        data_help = pprint.pformat(DataLoader.default_hparams().values())
        print("\nBase model default parameters: \n{}\n\n"
              "{} default parameters: \n{}\n\n"
              "{} data loader default parameters: \n{}".format(
                  base_help, args.model_name, specific_help, args.data_loader, data_help))
        return

    # parse the parameters, updating defaults
    model_hps = Model.parse_hparams(base=args.base_hparams, specific=args.hparams)
    data_hps = DataLoader.parse_hparams(args.data_hparams)

    # build model, load checkpoints
    utils.gpu.setup_gpu(args.gpu)
    dataset = DataLoader(data_hps, args.dataset)
    model = Model(model_hps, dataset, args.output_dir, args.id)
    if args.resume is not None:
        model.restore_checkpoint_if_exists(args.resume)

    # combine and save config file
    combined_hps = utils.hparams.combine_hparams_into_one(model_hps, data_hps)
    utils.hparams.save_config(model.config_filepath, combined_hps)

    # train
    model.train()


if __name__ == '__main__':
    main()
