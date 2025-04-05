---
# **Brain Age Prediction**
In this directory, we have codes for doing Brain Age Prediction. Here, we have one `*.py` file for merging and filtering needed information, one main `*.py` code for the brain age prediction, and one visualization code file.

## **Data**
As mentioned, we used [UK Biobank](https://www.ukbiobank.ac.uk/) cohort under Application ID: 41655 in our project. This cohort contains Neuroimaging scans (bulk data), Imaging derived phenotypes (IDPs), and Non-imaging derived phenotypes (non-IDPs). 

In this step, we used:
- ***T1-weighted MRI (which is a bulk Neuroimaging data)***
    - Using [*FAIRly big*](https://www.nature.com/articles/s41597-022-01163-2) [[1]](#references) and [*CAT 12*](https://academic.oup.com/gigascience/article/doi/10.1093/gigascience/giae049/7727520) [[2]](#references) we have calculated the Gray Matter Volume (GMV) of different parcels of the brain based on a combined Cortical ([Schaefer Cerebral Cortex parcellation atlas](https://academic.oup.com/cercor/article/28/9/3095/3978804)) [[3]](#references) and subcortical ([Melbourne Subcortex Atlas known as Tian atlas](https://www.nature.com/articles/s41593-020-00711-6)) [[4]](#references).

- ***Following non_IDPs***:
    - [Age](https://biobank.ndph.ox.ac.uk/showcase/field.cgi?id=21003)
    - [Sex](https://biobank.ndph.ox.ac.uk/showcase/field.cgi?id=31)
    - [ICD-10](https://biobank.ndph.ox.ac.uk/showcase/field.cgi?id=41202)
    - [Long-standing illness, disability or infirmity](https://biobank.ndph.ox.ac.uk/showcase/field.cgi?id=2188)
    - [Diabetes diagnosed by doctor](https://biobank.ndph.ox.ac.uk/showcase/field.cgi?id=2443)
    - [Age stroke diagnosed](https://biobank.ndph.ox.ac.uk/showcase/field.cgi?id=4056)
    - [Overall health rating](https://biobank.ndph.ox.ac.uk/showcase/field.cgi?id=2178)

## **Structure**
Using [*CAT 12*](https://academic.oup.com/gigascience/article/doi/10.1093/gigascience/giae049/7727520) [[2]](#references) we have calculated GMVs and stored them in the scratch sub directory:

```
📁 scratch
├── 📁 CATs
│   ├── Schaefer_17Network_200_UKB.csv
│   ├── Schaefer_17Network_400_UKB.csv
│   ├── Schaefer_17Network_600_UKB.csv
│   ├── Schaefer_17Network_800_UKB.csv
│   ├── Schaefer_17Network_1000_UKB.csv
│   ├── Tian_Subcortex_S1_UKB.csv
│   ├── Tian_Subcortex_S2_UKB.csv
│   ├── Tian_Subcortex_S3_UKB.csv
│   └── Tian_Subcortex_S4_UKB.csv
└── 📁 CATs_NoAge/(empty)

```
In `CATs` directory we have stored the results of [*CAT 12*](https://academic.oup.com/gigascience/article/doi/10.1093/gigascience/giae049/7727520) [[2]](#references) based on their granularities and atlas. After merging the cortical and subcortical atlases we save them in `CATs_NoAge` directory.



## Codes
1. `1_0_DataFiltering.py`
      - merging cortical and subcortical atlases
      - filtering participants into `Healthy` and `Rest of Population` based on [Cole](https://www.sciencedirect.com/science/article/pii/S0197458020301056) [[5]](#references) criteria using aforsaid non-IDPs.
      - saving the final data 
2. `1_1_BrainAgePrediction.py`
      - Training the models for predicting the Age of the brain using `Healthy` subset.
3. `1_2_visualisation.py`
      - Predicting the Age of the brain on `Rest of Population` subset
      - calculating the Brain Age GAP and correcting it for the bias
      - plotting and saving the performances


---
## **References**
1. Wagner, A.S., et al., FAIRly big: A framework for computationally reproducible processing of large-scale data. Scientific data, 2022. 9(1): p. 80.
2. Gaser, C., et al., CAT–A computational anatomy toolbox for the analysis of structural MRI data. biorxiv, 2022: p. 2022.06. 11.495736.
3. Schaefer, A., et al., Local-global parcellation of the human cerebral cortex from intrinsic functional connectivity MRI. Cerebral cortex, 2018. 28(9): p. 3095-3114.
4. Tian, Y., et al., Topographic organization of the human subcortex unveiled with functional connectivity gradients. Nature neuroscience, 2020. 23(11): p. 1421-1432.
5. Cole, J. H. (2020). Multimodality neuroimaging brain-age in UK biobank: relationship to biomedical, lifestyle, and cognitive factors. Neurobiology of aging, 92, 34-42.
