---
# **Brain Age Prediction**

In this directory, we have one main `*.py` code for the brain age prediction and one visulaisation code file.

## **Data**
As mentioned, we used [UK Biobank](https://www.ukbiobank.ac.uk/) cohort under Application ID: 41655 in our project. This cohort contains of Neuroimaging scans(bulk data), Imaging derived phenotypes (IDPs), and Non-imaging derived phenotypes (non-IDPs). 

In this step we used:
- ***T1-weighted MRI (which is a bulk Neuroimaging data)***
    - Using [*FAIRly big*](https://www.nature.com/articles/s41597-022-01163-2) [[1]](#references) and [*CAT 12*](https://academic.oup.com/gigascience/article/doi/10.1093/gigascience/giae049/7727520) [[2]](#references) we have calculated the Gray Matter Volume (GMV) of differenct parcels of the brain baed on a combined Cortical ([Schaefer Cerebral Cortex parcellation atlas](https://academic.oup.com/cercor/article/28/9/3095/3978804)) [[3]](#references) and subcortical ([Melbourne Subcortex Atlas known as Tian atlas](https://www.nature.com/articles/s41593-020-00711-6)) [[4]](#references).   

- ***Following non_IDPs***:
    - [Age](https://biobank.ndph.ox.ac.uk/showcase/field.cgi?id=21003)
    - [Sex](https://biobank.ndph.ox.ac.uk/showcase/field.cgi?id=31)
    - [ICD-10](https://biobank.ndph.ox.ac.uk/showcase/field.cgi?id=41202)
    - [Long-standing illness, disability or infirmity](https://biobank.ndph.ox.ac.uk/showcase/field.cgi?id=2188)
    - [Diabetes diagnosed by doctor](https://biobank.ndph.ox.ac.uk/showcase/field.cgi?id=2443)
    - [Age stroke diagnosed](https://biobank.ndph.ox.ac.uk/showcase/field.cgi?id=4056)
    - [Overall health rating](https://biobank.ndph.ox.ac.uk/showcase/field.cgi?id=2178)












---
## **References**
1. Wagner, A.S., et al., FAIRly big: A framework for computationally reproducible processing of large-scale data. Scientific data, 2022. 9(1): p. 80.
2. Gaser, C., et al., CAT–A computational anatomy toolbox for the analysis of structural MRI data. biorxiv, 2022: p. 2022.06. 11.495736.
3. Schaefer, A., et al., Local-global parcellation of the human cerebral cortex from intrinsic functional connectivity MRI. Cerebral cortex, 2018. 28(9): p. 3095-3114.
4. Tian, Y., et al., Topographic organization of the human subcortex unveiled with functional connectivity gradients. Nature neuroscience, 2020. 23(11): p. 1421-1432.
