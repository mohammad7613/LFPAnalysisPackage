# lfp_analysis/main.py
import sys
from lfp_analysis.config import load_config
from lfp_analysis.builder.yaml_builder import build_from_yaml

def main():
    # get config path from command line or default
    print("sys.argv:", sys.argv[1])
    print("len(sys.argv):", len(sys.argv))
    config_file = sys.argv[1] if len(sys.argv) > 1 else "/media/mohammad/atp/D/ComplexityCheckCodes/RawData/PiplineCodes/lfp_analysis/config/examples/te_analysis.yaml"
    print(f"Using config file: {config_file}")
    # build pipeline from yaml config
    pipeline, lfp = build_from_yaml(config_file)

    # run pipeline
    pipeline.run(lfp)

if __name__ == "__main__":
    main()


