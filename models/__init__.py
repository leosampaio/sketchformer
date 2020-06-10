import os
import pkgutil
import importlib

from core.models import BaseModel

models_by_name = {}
pkg_dir = os.path.dirname(__file__)
for (module_loader, name, ispkg) in pkgutil.iter_modules([pkg_dir]):
    importlib.import_module('.' + name, __package__)

all_subclasses = BaseModel.__subclasses__() + [s for ss in [s.__subclasses__() for s in BaseModel.__subclasses__()] for s in ss]
models_by_name = {cls.name: cls for cls in all_subclasses if hasattr(cls, 'name')}


def get_model_by_name(model_name):
    return models_by_name[model_name]
