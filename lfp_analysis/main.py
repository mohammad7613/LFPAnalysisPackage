# lfp_analysis/main.py
import sys
from lfp_analysis.builder.yaml_builder import build_from_yaml

from lfp_analysis.registry.autodiscovery import autodiscover

def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "D:\\D\\ComplexityCheckCodes\\RawData\\PiplineCodes\\lfp_analysis_git\\lfp_analysis\\config\\examples\\complexity_analysis_new.yaml"
        # --- make sure registry is populated ---
    autodiscover()
    
    pipeline = build_from_yaml(config_path)
    pipeline.summary()
    pipeline.run()

if __name__ == "__main__":
    main()




# # lfp_analysis/main.py
# import sys
# from lfp_analysis.config import load_config
# from lfp_analysis.builder.yaml_builder import build_from_yaml

# def main():
#     # get config path from command line or default
#     print("sys.argv:", sys.argv[1])
#     print("len(sys.argv):", len(sys.argv))
#     config_file = sys.argv[1] if len(sys.argv) > 1 else "/media/mohammad/atp/D/ComplexityCheckCodes/RawData/PiplineCodes/lfp_analysis/config/examples/te_analysis.yaml"
#     print(f"Using config file: {config_file}")
#     # build pipeline from yaml config
#     pipeline, lfp = build_from_yaml(config_file)

#     # run pipeline
#     pipeline.run(lfp)

# if __name__ == "__main__":
#     main()


