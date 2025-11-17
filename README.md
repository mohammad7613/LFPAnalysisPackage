# LFPAnalysisPackage

LFPAnalysisPackage is a domain-specific framework designed for analyzing Local Field Potential (LFP) data. It allows researchers to design and execute LFP analyses in a clear, high-level, and more comprehensible manner.

## Features

- Domain-specific language for LFP analysis
- Simplified syntax for common LFP workflows
- Modular and extensible design

## Installation

To install the package in editable mode, run in terminal in the folder where setup.py is located:

pip install -e .

## Execution

To excute your analysis, you need to run in the terminal

python -m lfp_analysis.main path_to_yaml_analysis


## Add new module

Steps:

1- Create a seprate folder with __init__.py and base.py(which includes the interfaces and abstract classes)
2- Import the module in autodiscovery.py in registery folder, follow the pattern by which the previous module has been added.
3- Use "from lfp_analysis.registry import register" in any file inside the folder(module) which contains which should be registered to be used in the pipeline
4- Design the run and build method in pipeline.py to involve your module appropriately
5- To create a suitable higher level yaml language for your module. The basic steps is to add dictionary key in in dict REGISTRIES in registry/base path. Then, create yaml section for this modulde. You need to create suitable higher level section pattern which match the lower level method calls and class initialization. Step 4 and 5 should be done simulateniously. 
6- run the corresponding yaml including your created language in main and check its execution and return to step 4 and 5 if there is any error.