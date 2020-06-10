import os
import pkgutil
import importlib

from core.experiments import Experiment

experiments_by_name = {}
pkg_dir = os.path.dirname(__file__)
for (module_loader, name, ispkg) in pkgutil.iter_modules([pkg_dir]):
    importlib.import_module('.' + name, __package__)

all_subclasses = Experiment.__subclasses__() + [s for ss in [s.__subclasses__() for s in Experiment.__subclasses__()] for s in ss]
experiments_by_name = {cls.name: cls for cls in all_subclasses if hasattr(cls, 'name')}


def get_experiment_by_name(exp_name):
    return experiments_by_name[exp_name]
