#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@authors: 
    Mostafa Mahdipour              m.mahdipour@fz-juelich.de
    Somayeh Maleki Balajoo         s.maleki.balajoo@fz-juelich.de
"""

# %%

import sys
import os
import numpy as np 
import pandas as pd # to use dataframe
# import psutil
from pathlib import Path
import datalad.api
import shutil
import nest_asyncio
nest_asyncio.apply()
# %% Color print in terminal 
RED   = "\033[1;31m"  
BLUE  = "\033[1;34m"
CYAN  = "\033[1;36m"
GREEN = "\033[0;32m"
RESET = "\033[0;0m"
BOLD    = "\033[;1m"
REVERSE = "\033[;7m"


UNDERLINE = '\033[4m'
CBLACKBG  = '\33[40m'
CREDBG    = '\33[41m'
CGREENBG  = '\33[42m'
CYELLOWBG = '\33[43m'
CBLUEBG   = '\33[44m'
CVIOLETBG = '\33[45m'
CBEIGEBG  = '\33[46m'
CWHITEBG  = '\33[47m'

# %% Starting statement
sys.stdout.write(RED)
# sys.stdout.write(BOLD)
print(
      "\n\n================================================="
      "\nprogram started\n"
      "=================================================\n\n"
      )
sys.stdout.write(RESET)
#%% Installing Datalad dataset, getting Demographic data copy to 1_DATA Folder
"""
If you want to download the newest version of UK Biobank accessible data-field file, 
 you should first create a suitable anaconda/miniconda environment and install the
 datalad and its other dependencies on your Juseless account. 
 Then, you can run the below commands to download this file:
     datalad clone "ria+http://ukb.ds.inm7.de#~bids" ukb_bids
     cd ukb_bids
     datalad get ukb670018.tsv

"""
My_current_path=os.path.abspath(os.getcwd())
DataLad_UKBB=My_current_path+'/path/to/your/datalad/ukb_bids'
Datalad_Path=Path(DataLad_UKBB)

dir = DataLad_UKBB
# create directory
if not os.path.exists(dir):    
    os.makedirs(dir)

dataset_url = ('ria+http://ukb.ds.inm7.de#~bids')
dataset = datalad.api.install(  # type: ignore
    path=Datalad_Path, source=dataset_url)

#getting the data
datalad.api.update(dataset=DataLad_UKBB,merge=True)

CSV_File=Path('ukb670018.tsv')

dataset.get(CSV_File)

#%% loading Demographic_Phenotypic file
# Loading UK Biobank's file
sys.stdout.write(CYAN)
print("\n Loading the UK Biobank Demographic/Phenotypic data : ukb670018.tsv")
sys.stdout.write(RESET)

#give data the access
os.chmod(Datalad_Path+'/ukb670018.tsv',0o776)

ukbsamp=pd.read_csv(Datalad_Path+'/ukb670018.tsv', sep='\t')

sys.stdout.write(CYAN)
print("\n The UK Biobank Demographic/Phenotypic data has been successfully loaded")
sys.stdout.write(RESET)
#%% loading CAT files
# laoding Scheafer and Tian Atlases and concatinating them into one dataframe

"""

based on our previous experience, it is more effective to use Schaefer atlas with Tian atlas in this order:
	100 parcels with S1
	200 parcels with S2
	300 parcels with S3
	400 and more parcels with S4
Therefore we merge them in this manner

"""

sys.stdout.write(CYAN)
print("\n Loading the UK Biobank cortical and sub-cortical data and concatinating them")
sys.stdout.write(RESET)

Path_to_data="Path/to/Data/" # It is an empty directory where you want to save the final data
# there you will have `CATs_Age`, `CATs_Age_Healthy`, and `Demographic_Phenotypic` folders. For now all of them are empty.

Path_to_scratch="Path/to/scratch/" # Imagine that you have a folder called scratch in your home directory and you have two sub directories there
# called CATs and CATs_NoAge .in the `CATs` you have the CAT 12 outputs based on Scheqefer atlas and Tian atlas seperatly (different granularity)
# `CATs_NoAge` is an empty directory where you will save merged Scheqefer atlas and Tian atlas dataframes (without participants's Ages)

for file in os.listdir(Path_to_scratch+'CATs/'):
    if file.startswith("Schaefer"):
        sys.stdout.write(CYAN)
        print(file)
        sys.stdout.write(RESET)
        
        Scheafer1=pd.read_csv(Path_to_scratch+'CATs/'+file,index_col=0).reset_index(drop=True)
        # print(Scheafer1.columns)
        if file.startswith("Schaefer_17Network_200_UKB"):
            Tian1=pd.read_csv(Path_to_scratch+'CATs/'+'Tian_Subcortex_S2_UKB.csv',index_col=0).reset_index(drop=True)
            # print(Tian1.columns)
            if Scheafer1['SubjectID'].equals(Tian1['SubjectID']) and Scheafer1['Session'].equals(Tian1['Session']):
                Scheafer_Tian=pd.concat([Scheafer1, Tian1.iloc[:,2:]], axis=1)
                #Tian1.iloc[:,3:], eliminates {"Unnamed: 0" "subject_ID" "session"} columns in Tian
                # print(Scheafer_Tian.columns)
                Scheafer_Tian.to_csv(Path_to_scratch+'CATs_NoAge/NoAge_'+file[:-4]+'_Tian.csv')
        else:
            Tian1=pd.read_csv(Path_to_scratch+'CATs/'+'Tian_Subcortex_S4_UKB.csv',index_col=0).reset_index(drop=True)
            # print(Tian1.columns)
            if Scheafer1['SubjectID'].equals(Tian1['SubjectID']) and Scheafer1['Session'].equals(Tian1['Session']):
                Scheafer_Tian=pd.concat([Scheafer1, Tian1.iloc[:,2:]], axis=1)
                #Tian1.iloc[:,3:], eliminates {"Unnamed: 0" "subject_ID" "session"} columns in Tian
                # print(Scheafer_Tian.columns)
                Scheafer_Tian.to_csv(Path_to_scratch+'CATs_NoAge/NoAge_'+file[:-4]+'_Tian.csv')

del(Tian1)
del(Scheafer1)
del(Scheafer_Tian)
sys.stdout.write(CYAN)
print("\n The UK Biobank cortical and sub-cortical data has been successfully loaded and concatinated")
sys.stdout.write(RESET)
#%% adding age and Sex to this list
sys.stdout.write(CYAN)
print("\n Starting to add age and Sex to the CAT files")
sys.stdout.write(RESET)


test1_noage=pd.read_csv(Path_to_scratch+'CATs_NoAge/'+'NoAge_Schaefer_17Network_200_UKB_Tian.csv',index_col=0).reset_index(drop=True) # only for extarcting
#subject whe have T1 MRI. 
Demo_all_PoP=ukbsamp[ukbsamp.eid.isin(test1_noage.SubjectID.str[4:].apply(int))].reset_index(drop=True) #phenotypic data for participants who have MRI scans
del(test1_noage)

for file in os.listdir(Path_to_scratch+'CATs_NoAge/'):
    if file != '.DS_Store':
        sys.stdout.write(CYAN)
        print(file)
        sys.stdout.write(RESET)
        test1_noage=pd.read_csv(Path_to_scratch+'CATs_NoAge/'+file,index_col=0).reset_index(drop=True)
        test1_noage.insert(2,'Age',np.nan)
        test1_noage.insert(3,'Sex',np.nan)
        if not(Demo_all_PoP.eid.equals(test1_noage.SubjectID.str[4:].apply(int))):
            sys.stdout.write(RED)
            print('ERROR insering Sex')
            sys.stdout.write(RESET)
        else :
            test1_noage.Sex=Demo_all_PoP['31-0.0']
            test1_noage.Age[test1_noage.Session=='ses-3']=Demo_all_PoP['21003-3.0'][test1_noage.Session=='ses-3']
            test1_noage.Age[test1_noage.Session=='ses-2']=Demo_all_PoP['21003-2.0'][test1_noage.Session=='ses-2']

        sys.stdout.write(CYAN)
        print(file)
        sys.stdout.write(RESET)
        test1_noage.to_csv(Path_to_data+'CATs_Age/'+file.replace('NoAge','Age'))

del(test1_noage)
del(file)
sys.stdout.write(CYAN)
print("\n Age and Sex have beed added to the CAT files")
sys.stdout.write(RESET)
#%% filtering


"""
Based on Cole's paper [https://www.sciencedirect.com/science/article/pii/S0197458020301056] 
 there are 5 data-fields which are :
     ICD-10                    41202
     long-standing illness     2188
     diabetes                  2443
     stroke                    4056
     not good health           2178

 The below commands will generate a dataframe of their very initial infromation
"""
HI_index_Dic={
    'ID':['ICD-10','long-standing illness','diabetes','stroke','Overall health rating']
               ,'ID_number':['41202','2188','2443','4056','2178']
               ,'coding':['19','100349','100349','100291','100508']}
ukbHI=pd.DataFrame(HI_index_Dic) #list of Coding IDs

sys.stdout.write(CYAN)
print("\n creating a dataframe of those Data-fields we want to filter data based on cole 2020 paper\n")
sys.stdout.write(RESET)

IDsforFilter1=ukbHI.ID_number.apply(str)+'-'
IDsforFilter=IDsforFilter1.tolist()
IDsforFilter.insert(0,'eid')
filtereddata=Demo_all_PoP[Demo_all_PoP.columns[pd.Series(Demo_all_PoP.columns).str.startswith(tuple(IDsforFilter))]]

sys.stdout.write(CYAN)
print("\n creating is finished")
sys.stdout.write(RESET)


sys.stdout.write(CYAN)
print("\n Filtering data based on cole criteria has been started")
sys.stdout.write(RESET)




"""
after creating sub-DataFrame from UK BioBank Demographic_Phenotypic to only have Cole criteria's Field-ID,
We can now start extract Super Healthy participantc who also have MRI scans.

"""



field_id='41202'
#the healthy participant shouldn't have any diagnosis
ICD_10_init=filtereddata.iloc[:,filtereddata.columns.str.startswith(field_id)]
ICD_10=ICD_10_init[ICD_10_init.isna().all(axis=1)]


field_id='2188'
long_standing_illness=filtereddata.iloc[ICD_10.index,filtereddata.columns.str.startswith(field_id)]
#long_standing_illness is coded by coding 100349, each element can have these values:
#1	Yes
#0	No
#-1	Do not know
#-3	Prefer not to answer
#or NaN

#long_standing_illness has four columns for each participant. representing four time instances
#at least one of columns should be 0 so we can undestand the paticipant doesn't have long standing illness
# 0     0       0       0
# 0     0       0       NaN
# 0     0       NaN     0
# 0     NaN     0       0
# NaN   0       0       0
# ...(there are other permutation possibilities that we assume you know all others)
# 0     NaN     NaN     NaN
# NaN   0       NaN     NaN
# NaN   NaN     0       NaN
# NaN   NaN     NaN     0
#(in this case that at least one element is 0, other elements should be null 
# this assumption automatically be considered when we add other logical conditions)

# if all elenments are NaN it is not acceptable
#  NaN   NaN     NaN     NaN ==> not acceptable

# if any of elements be -3 that means the person didn't
# want to aswer. So there is this possibilty that the person
# had long-standing illnes but didn't want to tell.

# From a strictly conservative perspective, if the participant has
# answered, at even one of the time instances, "yes, I have a long-standing
#illness," then we should exclude that person.
# examples:
# NaN     1      0       NaN     ==> exclude
# 0       0      1       0       ==> exclude
ICD_10_LSI=long_standing_illness[(long_standing_illness.values==0).any(axis=1) & ~(long_standing_illness.values==-3).any(axis=1) & ~(long_standing_illness.values==-1).any(axis=1) & ~(long_standing_illness.values==1).any(axis=1)]



field_id='2443'
diabetes=filtereddata.iloc[ICD_10_LSI.index,filtereddata.columns.str.startswith(field_id)]
#diabetes coding is same as long-standing illness so the filtering criteria would be the same
ICD_10_LSI_DB=diabetes[(diabetes.values==0).any(axis=1) & ~(diabetes.values==-3).any(axis=1) & ~(diabetes.values==-1).any(axis=1) & ~(diabetes.values==1).any(axis=1)]


field_id='4056'
stroke=filtereddata.iloc[ICD_10_LSI_DB.index,filtereddata.columns.str.startswith(field_id)]

#Stroke has been coded by coding 100291, each element can have these values:
#-1	Do not know
#-3	Prefer not to answer
#Stroke has four columns for each participant. representing four time instances
#all of them should be NaN because of the coding

## correct the doc
ICD_10_LSI_DB_ST=stroke[stroke.isna().all(axis=1)]


field_id='2178'
goodhealth=filtereddata.iloc[ICD_10_LSI_DB_ST.index,filtereddata.columns.str.startswith(field_id)]
#Overall health rating has been coded by coding 100508, each element can have these values:
# 1	Excellent
# 2	Good
# 3	Fair
# 4	Poor
# -1	Do not know
# -3	Prefer not to answer

#Overall health rating has four columns for each participant. representing four time instances
#all of them should be NaN because of the coding

# if all elenments are NaN it is not acceptable
#  NaN   NaN     NaN     NaN ==> not acceptable
# if any of elements be -3 that means the person didn't
# want to aswer. So there is this possibilty that the person
# had not overall good health and because of that he/she didn't want to tell.
# So we see that answer even one time, we should exclude
#if the person does not know, again, there is
#the possibility of having not overall good health 
# rejection condition (paranoid strict conservative criteria)
# example:
# NaN     4      1       NaN     ==> exclude
# 1       2      4       3       ==> exclude

ICD_10_LSI_DB_ST_GH=goodhealth[~(goodhealth.isna().all(axis=1)) & ~(goodhealth.values==4).any(axis=1) & ~(goodhealth.values==-3).any(axis=1) & ~(goodhealth.values==-1).any(axis=1)]





FD_Sup_Healthy=Demo_all_PoP.iloc[ICD_10_LSI_DB_ST_GH.index,:]


Demo_all_PoP.to_csv(Path_to_data+'Demographic_Phenotypic/'+'Demographic_Phenotypic_MRI_PoP.csv') # Demographic of participants
# who have MRI data
FD_Sup_Healthy.to_csv(Path_to_data+'Demographic_Phenotypic/'+'Demographic_Phenotypic__MRI_Super_Healthy.csv') # Demographic
#of participants who have MRI data and filtered as Super Healthy subjects

del(HI_index_Dic)
del(IDsforFilter1)
del(IDsforFilter)
del(filtereddata)
del(field_id) 
del(ICD_10_init)         
del(ICD_10)
del(long_standing_illness)
del(ICD_10_LSI)
del(diabetes)
del(ICD_10_LSI_DB)
del(stroke)
del(ICD_10_LSI_DB_ST)
del(goodhealth)
# del(ICD_10_LSI_DB_ST_GH)

sys.stdout.write(CYAN)
print("\n Filtering data based on cole criteria has been Finished")
sys.stdout.write(RESET)
#%% Creating CAT csv files for Super Healthy participants
sys.stdout.write(CYAN)
print("\n Starting to create *.csv files for Super Healthy participants")
sys.stdout.write(RESET)




for file in os.listdir(Path_to_data+'CATs_Age/'):
    if file != '.DS_Store':
        sys.stdout.write(CYAN)
        print(file)
        sys.stdout.write(RESET)
        Scheafer_Tian=pd.read_csv(Path_to_data+'CATs_Age/'+file,index_col=0).reset_index(drop=True)
        Scheafer_Tian_Healthy=Scheafer_Tian[Scheafer_Tian.SubjectID.isin('sub-'+FD_Sup_Healthy.eid.apply(str))]
        Scheafer_Tian_Healthy.to_csv(Path_to_data+'CATs_Age_Healthy/'+file.replace('Tian','Tian_Healthy'))
        sys.stdout.write(CYAN)
        print(Path_to_data+'CATs_Age_Healthy/'+file.replace('Tian','Tian_Healthy'))
        sys.stdout.write(RESET)
        
del(Scheafer_Tian)
del(file)
sys.stdout.write(CYAN)
print("\n Creating *.csv files for Super Healthy participants has been finished")
sys.stdout.write(RESET)

# %% Ending statement
sys.stdout.write(RED)

print(
      "\n\n================================================="
      "\nprogram ended\nenjoy ;)\n"
      "=================================================\n\n"
      )
sys.stdout.write(RESET)
# %%
