# lfp_analysis/registry/autodiscover.py
import importlib
import pkgutil
import lfp_analysis.feature
import lfp_analysis.preprocess
import lfp_analysis.visualization

def autodiscover():
    """
    Automatically import all submodules in the main component packages
    so that their @register decorators run and populate REGISTRIES.
    """
    packages = [
        lfp_analysis.feature,
        lfp_analysis.preprocess,
        lfp_analysis.visualization,
    ]

    for pkg in packages:
        for _, module_name, _ in pkgutil.iter_modules(pkg.__path__):
            importlib.import_module(f"{pkg.__name__}.{module_name}")
