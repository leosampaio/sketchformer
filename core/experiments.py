import os
from abc import ABCMeta, abstractmethod

import utils
from core.notifyier import Notifyier


class Experiment(object, metaclass=ABCMeta):
    """Abstract base class for metrics/experiments that take a long time to
    run or that do not need to follow the SlowMetrics protocol either for 
    not ever being computed during training or because they do not require a
    trained model. If they do require a loaded model, that is should be 
    advertised in the class attribute requires_model. Children should also 
    have a name attribute for dynamic loading on the experiment.py script

    Experiments also have their own HParams, just making them even more fun
    to use
    """

    @classmethod
    def base_default_hparams(cls):
        base_hparams = utils.hparams.HParams(
            slack_config='token.secret',  # file with slack setup (token and channel)
        )
        return base_hparams

    def __init__(self, hps, experiment_sub_id, outdir):
        self.hps = hps if isinstance(hps, dict) else dict(hps.values())
        self.identifier = "{}-{}".format(self.name, experiment_sub_id)

        if not hasattr(self, 'name'):
            raise Exception("You must give your experiment a reference name")
        if not hasattr(self, 'requires_model'):
            raise Exception("You must advertise if your experiment requires "
                            "a trained model")

        self.out_dir = os.path.join(outdir, self.identifier)
        if not os.path.isdir(self.out_dir):
            os.mkdir(self.out_dir)

        self.notifyier = Notifyier(self.hps['slack_config'])

    @classmethod
    def default_hparams(cls):
        """Returns the default hparams for this model 
        """
        return utils.hparams.combine_hparams_into_one(
            cls.specific_default_hparams(),
            cls.base_default_hparams())

    @classmethod
    def parse_hparams(cls, new_hps):
        """Overrides both default sets of parameters (base and specific) and
        combines them into a single HParams objects, which is returned to 
        the caller.

        It is done this way so that the caller can save those parameters
        however they want to
        """
        hps = cls.default_hparams()
        if new_hps is not None:
            hps = hps.parse(new_hps)
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
    def compute(self, model=None):
        pass
