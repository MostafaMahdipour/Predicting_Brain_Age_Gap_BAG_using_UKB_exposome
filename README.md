
# Brain Age Prediction

This project aims first to predict brain age using pre/processed neuroimaging data as inputs and Machine Learning (ML) methods, then calculate Brain Age Gap (BAG), and finally investigate the relationship between the BAG and Demographic/phenotypic measurements of participants.
## Installation

Making a virtual environment:

Go to https://docs.conda.io

Install Miniconda (command line installer)

* In your Laptop/PC
* In Juseless

Open two terminals, one on your laptop and one in Juseless. Type the commands below:

create env:
```bash
conda create -n BrainAge python=3.9
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
install nilearn
```bash
conda install -c conda-forge nilearn
```
install Julearn 
```bash
pip install -U julearn
```
install Seaborn
```bash
conda install -c conda-forge seaborn
```
install scikit-learn
```bash
conda install -c conda-forge scikit-learn
```
install ipykernel (for working in VScode)
```bash
conda install ipykernel
```
Install git-anext
```bash
conda install -c conda-forge git-annex
```
Then you can install Spyder for the virtual environment to run the code.
Alternatively, you can open Spyder IDE or Visual Studio Code (CSV) and change their virtual environment from `base` to the `BrainAge` and easily work with code.

* Recomandation :
In case you want to work on Juseless it is highly recomended to use VSC with its `remote explorer`
## Directory Structure
There are several subdirectories in the main directory.

1.   `1_DATA`: the principal data we don't change again has been stored here. There are also some subdirectories in this folder:

* `Demographic_Phenotypic` : Which contains the Demographic/Phenotypic `*.csv` files from UK Biobank and their filtered `*.csv` files.

* `CATs`: Separate `*.csv` files derived from CAT toolbox for cortical (Sheafer Atlas with different parcels) and sub-cortical (Tian Atlas with different parcels) Atlases which contains the neuroimaging features. 

* `CATs_Age`: Contains merged cortical and subcortical atlases files, together with age and sex of them.

* `CATs_Age_Healthy`: Same as `CATs_Age` folder but only contains super healthy participants.

2. `2_src`: The codes

3. `3_scratch`: files downloaded from datalad, test, no needed files

4. `4_Results`: we have saved our results here.


## Usage/Examples
* In personal devide:
    * open a terminal in the directoy and type:
```bash
conda activate BrainAge
./BrainAge.sh
```
* In Juseless:
    * open a terminal in the directoy and type:
```bash
conda activate BrainAge
condor_submit BrainAge.submit
```

## Logos
* Forschungszentrum Jülich
![Logo](https://www.fz-juelich.de/SharedDocs/Bilder/INM/INM-1/EN/FZj_Logo.jpg;jsessionid=055504225E9296ADA3087AA0705C2529?__blob=poster)
* Institute of Neuroscience and Medicine
    * Brain and Behaviour (INM-7)
![Logo](https://www.fz-juelich.de/SharedDocs/Bilder/INM/INM-7/EN/Verschiedenes/Startbild.png?__blob=normal)
* Cognitive Neuroinformatics Group
![Logo](https://www.fz-juelich.de/SharedDocs/Bilder/INM/INM-7/EN/Cognitive%20Neuroinformatics.png?__blob=wide)



## Authors

- [@Mostafa Madipour](https://github.com/MostafaMahdipour)

