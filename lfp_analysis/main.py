# lfp_analysis/main.py
import sys
from lfp_analysis.builder.yaml_builder import build_from_yaml

from lfp_analysis.registry.autodiscovery import autodiscover


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "/media/mohammad/atp/D/ComplexityCheckCodes/RawData/PiplineCodes/lfp_analysis_git/lfp_analysis/config/examples/complexity_analysis_new.yaml"
        # --- make sure registry is populated ---
    autodiscover()
    
    pipeline = build_from_yaml(config_path)
    pipeline.summary()
    pipeline.run()

if __name__ == "__main__":
    main()







