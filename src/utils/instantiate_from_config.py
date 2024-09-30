import importlib

def instantiate_from_config(config,extra_kwargs={}):
    if not 'python_class' in config:
        raise KeyError('Expected key `python_class` to instantiate.')
    module, cls = config.python_class.rsplit('.', 1)
    kwargs = config.get('params',{}) or {}
    kwargs.update(extra_kwargs)
    return getattr(importlib.import_module(module, package=None), cls)(**kwargs)
