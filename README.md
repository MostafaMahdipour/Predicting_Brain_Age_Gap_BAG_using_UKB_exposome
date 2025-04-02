# **What Predicts Individual Brain Health?**

In this project, we aimed first to predict brain age using T1-weighted MRI scans. We leveraged the richness of a well-known large cohort of (cognitively) healthy participants in the UK [Biobank](https://www.ukbiobank.ac.uk/) to develop machine learning (ML) models that predict brain age from structural MRI data (known as imaging variables), and then tested these models on the rest of the UK Biobank population.

Next, we calculated the gap between predicted brain age and chronological age — known as the **Brain Age Gap (BAG)**. BAG is thought to serve as an important biomarker reflecting pathological processes in the brain.

Subsequently, after calculating BAG, we aimed to characterize it in relation to a range of demographic, biomedical, lifestyle, and other variables (i.e., exposomes). We again utilized state-of-the-art ML models to characterize BAG using the aforementioned exposomes.

---

## 📌 Project Structure

This project consists of two main steps:

- **Brain Age Prediction**
- **BAG Characterization**

Each of these steps has a separate sub-directory with a small `README.md` file describing its content.

---


## ⚙️ Prerequisites

Below, we describe the prerequisites for running the code.
The frist step is to create a conda environment so we can 
### Installation

**Recomendation**: Making a virtual environment

Go to https://docs.conda.io

Install Miniconda (command line installer)

Open a terminal and type the commands below:

create [env](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html):
```bash
conda create -n BrainAge python=3.11.3
```
activating the environment:
```bash
conda activate BrainAge
```
install ipython to the newly created environment
```bash
conda install ipython
```
install datalad
```bash
conda install -c conda-forge datalad
```
install Julearn. For Julearn do one of these three options (for more information check thier [webpage](https://juaml.github.io/julearn/main/index.html)): 
```bash
pip install -U julearn
pip install -U julearn [viz]
conda install -c conda-forge julearn
``` 
install Seaborn
```bash
conda install -c conda-forge seaborn
```
install scikit-learn
```bash
conda install -c conda-forge scikit-learn
```
install optuna
```bash
conda install -c conda-forge optuna
```
install SHAP. This package is for visualazing and inteperting the results of predictions. More informationon their [webpage](https://shap.readthedocs.io/en/latest/index.html)
```bash
conda install -c conda-forge shap
```

install pickle.

To install this package run one of the following:
```bash
conda install conda-forge::pickle5
conda install conda-forge/label/cf201901::pickle5
conda install conda-forge/label/cf202003::pickle5
conda install conda-forge/label/gcc7::pickle5
```

---
* ***Other recomandations:***

Install git-anext
```bash
conda install -c conda-forge git-annex
```
Install pickle.

To install this package run one of the following:
```bash
conda install conda-forge::pickle5
conda install conda-forge/label/cf201901::pickle5
conda install conda-forge/label/cf202003::pickle5
conda install conda-forge/label/gcc7::pickle5
```
install ipykernel (for working in VScode)
```bash
conda install ipykernel
```
install missingno
```bash
conda install -c conda-forge missingno
```
install pingouin
```bash
conda install -c conda-forge pingouin
```

install nilearn
```bash
conda install -c conda-forge nilearn
```
install nibabel
```bash
conda install conda-forge::nibabel
```
install click
```bash
conda install -c anaconda click
```
install wget (use one of the commands below)
```bash
conda install -c anaconda wget
pip install wget
```
install natsort
```bash
conda install anaconda::natsort
```
install jupyter
```bash
conda install jupyter
```
install panel
```bash
conda install panel
```
install pyviz_comms
```bash
conda install pyviz_comms
```
Then you can install Spyder for the virtual environment to run the code.
Alternatively, you can open Spyder IDE or Visual Studio Code (CSV) and change their virtual environment from `base` to the `BrainAge` and easily work with code.


### Directory Structure
As mentioned before, here are two main subdirectories here.

1. `BrianAgePrediction`: Code and explaination for Brain Age Prediction

2. `BAG_Prediction` : Code and explaination for Brain Age Prediction



## Authors

- [@Mostafa Madipour](https://github.com/MostafaMahdipour)