import metrics
import argparse
import pprint

import utils
import models
import dataloaders
import experiments


def main():

    parser = argparse.ArgumentParser(
        description='Train modified transformer with sketch data')
    parser.add_argument("experiment_name", default=None,
                        help="Reference name of experiment that you want to run")
    parser.add_argument("--id", default="0", help="Experiment signature")

    parser.add_argument("-o", "--output-dir", default="", help="output directory")
    parser.add_argument("--exp-hparams", default=None,
                        help="Parameters to override defaults for experiment")
    parser.add_argument("--model-hparams", default=None,
                        help="Parameters to override defaults for model")
    parser.add_argument("-g", "--gpu", default=0, type=int, help="GPU ID to run on", )

    parser.add_argument("--model-name", default=None,
                        help="Model that ou want to experiment on")
    parser.add_argument("--model-id", default=None,
                        help="Model that ou want to experiment on")

    parser.add_argument("--data-loader", default='stroke3-distributed',
                        help="Data loader that will provide data for model, "
                        "if you want to load a model")
    parser.add_argument("--dataset", default=None,
                        help="Input data folder if you want to load a model")

    parser.add_argument("--help-hps", action="store_true",
                        help="Prints out the hparams default values")
    args = parser.parse_args()

    Experiment = experiments.get_experiment_by_name(args.experiment_name)

    # check for lost users in the well of despair
    if args.help_hps:
        hps_description = pprint.pformat(Experiment.default_hparams().values())
        print("\nDefault params for experiment {}: \n{}\n\n".format(
            args.experiment_name, hps_description))
        return

    # load model if that is what the experiment requires
    utils.gpu.setup_gpu(args.gpu)
    if Experiment.requires_model:
        Model = models.get_model_by_name(args.model_name)
        DataLoader = dataloaders.get_dataloader_by_name(args.data_loader)

        # load the modelconfig
        model_hps = utils.hparams.combine_hparams_into_one(
            Model.default_hparams(), DataLoader.default_hparams())
        utils.hparams.load_config(
            model_hps, Model.get_config_filepath(args.output_dir, args.model_id))

        # optional override of parameters
        if args.model_hparams:
            model_hps.parse(args.model_hparams)

        dataset = DataLoader(model_hps, args.dataset)
        model = Model(model_hps, dataset, args.output_dir, args.model_id)
        model.restore_checkpoint_if_exists()
    else:
        dataset, model = None, None

    experiment_hps = Experiment.parse_hparams(args.exp_hparams)

    # finally, run the experiment
    experiment = Experiment(experiment_hps, args.id, args.output_dir)
    experiment.compute(model)


if __name__ == '__main__':
    main()
