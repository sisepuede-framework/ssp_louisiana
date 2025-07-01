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

### Getting Started

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

   * The SISEPUEDE output data is stored in an S3 bucket. Therefore, it is important to set up your AWS SSO login credentials to access the bucket through the AWS CLI. Please follow this [tutorial](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-sso.html#sso-configure-profile-token-auto-sso) if you haven't set your credentials.
   * You can also refer to the tutorial in [metamodel/notebooks/aws\_test.ipynb](metamodel/notebooks/aws_test.ipynb) to learn how to properly retrieve the data.

### Important Files

* [metamodel/notebooks/etl_4.1.ipynb](metamodel/notebooks/etl_4.1.ipynb): Notebook where the `lhs_samples` data, the SISEPUEDE emission output data and the Cost-Benefit data are merged. The training dataframe is created here.
* [metamodel/notebooks/model\_draft\_gb\_5.1.ipynb](metamodel/notebooks/model_draft_gb_5.1.ipynb): Notebook containing the machine learning pipeline for training a gradient boosting model to predict multiple targets.
* [metamodel/notebooks/utils/eda\_utils.py](metamodel/notebooks/utils/eda_utils.py): Utility module for exploratory data analysis and data cleaning.
* [metamodel/notebooks/utils/ml\_utils\_v2.py](notebooks/utils/ml_utils_v2.py): Utility module for machine learning pipelines.
* [metamodel/notebooks/trained_models/xgb_pipeline_5.1.pkl](metamodel/notebooks/trained_models/xgb_pipeline_5.1.pkl): Trained ML model that can be loaded in python and used to predict multitargets with new observations.