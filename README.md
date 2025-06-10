# Proposal for the repository
## Folders in the basic structure
* .config: Configurations necessary for a local machine
* data: Here we could put the data pipelines v1 and v2
* doc: Documentation
* model: src of the model, i.e. the sisepuede repository
* samples: code that helps get started with the model and supports the documentation
* usecases: here we can put the Louisiana code 
* test: Unit and integration tests or general model tests

## FILES IN THE STRUCTURE
* README: Introduction and presentation to the project
* LICENSE: explains the rights, restrictions, regulations and other legal considerations
* CHANGELOG - a documented history of important versions and updates
* CONTRIBUTORS: List of people who contribute to the repository
* AUTHORS: Most significant people of the project.

## Metamodel

Metamodel = Surrogate ML model + Optimization routine

## Getting Started

To get started, follow these steps:

1. **Create a new Conda environment**:

   ```bash
   conda create -n ssp_metamodel_env python=3.11
   conda activate ssp_metamodel_env
   ```

2. **Install the required dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure you have the SISEPUEDE output data**:

   * Under `metamodel/data/`, create a folder named `ssp`.
   * Inside the `ssp` folder, place a SISEPUEDE output database folder. You can obtain it from this [OneDrive link](https://tecmx-my.sharepoint.com/:f:/g/personal/juan_robledo_tec_mx/EiFqmgg61l9GlkMyKiF57bEB4736Kw5_VRhYlt34z2bZoA?e=FiTL6X).
   * *Note*: This is temporary. We should improve the codebase to allow direct access to these files from AWS S3.

You're now ready to use the Metamodel!

## Important Files

* [metamodel/notebooks/etl.ipynb](metamodel/notebooks/etl.ipynb): Notebook where the `lhs_samples` data and the SISEPUEDE emission output data are merged. The training dataframe is created here.
* [metamodel/notebooks/model\_draft\_gb\_2.ipynb](metamodel/notebooks/model_draft_gb_2.ipynb): Notebook containing the machine learning pipeline for training a gradient boosting model to predict emissions per subsector.
* [metamodel/notebooks/utils/eda\_utils.py](metamodel/notebooks/utils/eda_utils.py): Utility module for exploratory data analysis and data cleaning.
* [metamodel/notebooks/utils/ml\_utils.py](notebooks/utils/ml_utils.py): Utility module for machine learning pipelines.