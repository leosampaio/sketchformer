import metrics
import argparse
import pprint

import utils
import models
import dataloaders


def main():

    parser = argparse.ArgumentParser(
        description='Train modified transformer with sketch data')
    parser.add_argument("model_name", default=None, help="Model that we are going to train")
    parser.add_argument("--id", default="0", help="experiment signature")
    parser.add_argument("--data-loader", default='stroke3-distributed',
                        help="Data loader that will provide data for model")
    parser.add_argument("--dataset", default=None, help="Input data folder")
    parser.add_argument("-o", "--output-dir", default="", help="output directory")
    parser.add_argument('-p', "--hparams", default=None,
                        help="Parameters to override")
    parser.add_argument("-g", "--gpu", default=0, type=int, nargs='+', help="GPU ID to run on", )
    parser.add_argument('--metrics', type=str, nargs='+',
                        help="selection of metrics you want to calculate")
    parser.add_argument("--help-hps", action="store_true",
                        help="Prints out the hparams file")
    parser.add_argument("-r", "--resume", default='latest', help="One of 'latest' or a checkpoint name")
    args = parser.parse_args()

    # get our model and data loader classes
    Model = models.get_model_by_name(args.model_name)
    DataLoader = dataloaders.get_dataloader_by_name(args.data_loader)

    # load the config
    hps = utils.hparams.combine_hparams_into_one(Model.default_hparams(),
                                                 DataLoader.default_hparams())
    utils.hparams.load_config(hps, Model.get_config_filepath(args.output_dir, args.id))

    # check for help screams from the void
    if args.help_hps:
        combined_hps = pprint.pformat(hps.values())
        print("\nLoaded parameters from {}: \n{}\n\n".format(
            args.model_dir, combined_hps))
        return

    # optional override of parameters
    if args.hparams:
        hps.parse(args.hparams)

    # build model, load checkpoints
    utils.gpu.setup_gpu(args.gpu)
    dataset = DataLoader(hps, args.dataset)
    model = Model(hps, dataset, args.output_dir, args.id)
    model.restore_checkpoint_if_exists(args.resume)

    # compute and send metrics
    metric_names = args.metrics
    metrics_list = {m: metrics.build_metric_by_name(m, hps.values()) for m in metric_names}
    model.compute_metrics_from(metrics_list)
    model.plot_and_send_notification_for(metrics_list)
    model.clean_up_tmp_dir()

if __name__ == '__main__':
    main()
