# lfp_analysis/main.py
import sys
from lfp_analysis.builder.yaml_builder import build_from_yaml

from lfp_analysis.registry.autodiscovery import autodiscover


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "/home/mohammad/Desktop/PiplineCodes/lfp_analysis_git/lfp_analysis/config/examples/Figure1_examplePower.yaml"
        # --- make sure registry is populated ---Figure1_exampleـpower
    cash_path = sys.argv[2] if len(sys.argv) > 2 else "/home/mohammad/Desktop/PiplineCodes/lfp_analysis_git/lfp_analysis/cachfiles/"
    autodiscover()
    pipeline = build_from_yaml(config_path=config_path, cache_path=cash_path)
    pipeline.summary()
    pipeline.run()

if __name__ == "__main__":
    main()







