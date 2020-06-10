import os
import pkgutil
import importlib

from core.metrics import SlowMetric

metrics_by_name = {}
pkg_dir = os.path.dirname(__file__)
for (module_loader, name, ispkg) in pkgutil.iter_modules([pkg_dir]):
    importlib.import_module('.' + name, __package__)

all_subclasses = SlowMetric.__subclasses__() + [s for ss in [s.__subclasses__() for s in SlowMetric.__subclasses__()] for s in ss]
metrics_by_name = {cls.name: cls for cls in all_subclasses if hasattr(cls, 'name')}


def build_metric_by_name(metric_name, params):
    return metrics_by_name[metric_name](params)
