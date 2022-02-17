#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: Mostafa Mahdipour
"""

# %%

import sys
import os
import numpy as np 
import pandas as pd # to use dataframe
import psutil

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
#%% loading Demographic_Phenotypic file
# Loading UK Biobank's file
"""
If you want to download the newest version of UK Biobank accessible data-field file, 
 you should first create a suitable anaconda/miniconda environment and install the
 datalad and its other dependencies on your Juseless account. 
 Then, you can run the below commands to download this file:
     datalad clone "ria+http://ukb.ds.inm7.de#~bids" ukb_bids
     cd ukb_bids
     datalad get ukb45132.tsv

"""
sys.stdout.write(CYAN)
print("\n Loading the UK Biobank Demographic/Phenotypic data : ukb45132.tsv")
sys.stdout.write(RESET)
process = psutil.Process(os.getpid())
print(process.memory_info().rss)  # in bytes 
# ukbsamp=pd.read_csv('../1_DATA/Demographic_Phenotypic/ukb45132.tsv', sep='\t')
ukbsamp=pd.read_csv('../1_DATA/Demographic_Phenotypic/ukb45132.csv')
process = psutil.Process(os.getpid())
print(process.memory_info().rss)  # in bytes 

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

Path_to_data="../1_DATA/CATs/"
Path_to_scratch='../3_scratch/1_DataFiltering/'

for file in os.listdir(Path_to_data):
    if file.startswith("Schaefer"):
        print(file)
        Scheafer1=pd.read_csv(Path_to_data+file)
        # print(Scheafer1.columns)
        if file.startswith("Schaefer_17Network_200_UKB"):
            Tian1=pd.read_csv(Path_to_data+'Tian_Subcortex_S2_UKB.csv')
            # print(Tian1.columns)
            if Scheafer1['subject_ID'].equals(Tian1['subject_ID']) and Scheafer1['session'].equals(Tian1['session']):
                Scheafer_Tian=pd.concat([Scheafer1, Tian1.iloc[:,3:]], axis=1)
                #Tian1.iloc[:,3:], eliminates {"Unnamed: 0" "subject_ID" "session"} columns in Tian
                # print(Scheafer_Tian.columns)
                Scheafer_Tian.to_csv(Path_to_scratch+'NoAge_'+file[:-4]+'_Tian.csv')
        else:
            Tian1=pd.read_csv(Path_to_data+'Tian_Subcortex_S4_UKB.csv')
            # print(Tian1.columns)
            if Scheafer1['subject_ID'].equals(Tian1['subject_ID']) and Scheafer1['session'].equals(Tian1['session']):
                Scheafer_Tian=pd.concat([Scheafer1, Tian1.iloc[:,3:]], axis=1)
                #Tian1.iloc[:,3:], eliminates {"Unnamed: 0" "subject_ID" "session"} columns in Tian
                # print(Scheafer_Tian.columns)
                Scheafer_Tian.to_csv(Path_to_scratch+'NoAge_'+file[:-4]+'_Tian.csv')


sys.stdout.write(CYAN)
print("\n The UK Biobank cortical and sub-cortical data has been successfully loaded and concatinated")
sys.stdout.write(RESET)
#%% adding age and gender to this list
sys.stdout.write(CYAN)
print("\n Starting to add age and gender to the CAT files")
sys.stdout.write(RESET)

for file in os.listdir(Path_to_scratch):
    print(file)
    Scheafer_Tian=pd.read_csv(Path_to_scratch+file)
    Scheafer_Tian.drop(columns=['Unnamed: 0','Unnamed: 0.1'],inplace=True)
    Scheafer_Tian.insert(2,'Gender',np.nan)
    Scheafer_Tian.insert(2,'Age',np.nan)
    for i in np.arange(Scheafer_Tian.shape[0]):
        jj=ukbsamp.eid[ukbsamp.eid==Scheafer_Tian.subject_ID.str[4:].astype(int)[Scheafer_Tian.index[i]]].index[0]
        if Scheafer_Tian.iloc[i,Scheafer_Tian.columns.get_loc('session')]=='ses-2':
            Scheafer_Tian.iloc[i,Scheafer_Tian.columns.get_loc('Age')]=ukbsamp.iloc[jj,ukbsamp.columns.get_loc('21003-2.0')]
            Scheafer_Tian.iloc[i,Scheafer_Tian.columns.get_loc('Gender')]=ukbsamp.iloc[jj,ukbsamp.columns.get_loc('31-0.0')]
        else:
            Scheafer_Tian.iloc[i,Scheafer_Tian.columns.get_loc('Age')]=ukbsamp.iloc[jj,ukbsamp.columns.get_loc('21003-3.0')]
            Scheafer_Tian.iloc[i,Scheafer_Tian.columns.get_loc('Gender')]=ukbsamp.iloc[jj,ukbsamp.columns.get_loc('31-0.0')]
    Scheafer_Tian.to_csv(Path_to_data.replace('CATs','CATs_Age')+file.replace('NoAge','Age'))

sys.stdout.write(CYAN)
print("\n Age and gender have beed added to the CAT files")
sys.stdout.write(RESET)
#%% filtering
MRI_list_subs=Scheafer_Tian.subject_ID.str[4:] # list of participant IDs who have MRI data
Demo_all_PoP=ukbsamp[ukbsamp.eid.isin(MRI_list_subs)] #phenotypic data for participants who have MRI scans


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
HI_index_list={
    'ID':['ICD-10','long-standing illness','diabetes','stroke','not good health']
               ,'ID_number':['41202','2188','2443','4056','2178']
               ,'coding':['19','100349','100349','100291','100508']}
ukbHI=pd.DataFrame(HI_index_list) #list of Coding IDs

sys.stdout.write(CYAN)
print("\n creating a dataframe of those Data-fields we want to filter data based on cole 2020 paper")
sys.stdout.write(RESET)

listcolumns=list(ukbsamp.columns) # list of columns in Demographic
d=[0] # index for cole criteria
s=['eid']
[rows1,cols1]=Demo_all_PoP.shape
[rows2,cols2]=ukbHI.shape

# creating a dataframe of only Data-fields we want to look at
# each data Field has at least 4 columns for 4 different time instances
for rowHID in np.arange(rows2):
    zz=len(ukbHI.ID_number[rowHID])
        
    for col_index in np.arange(cols1):
            if listcolumns[col_index][0:zz]==ukbHI.ID_number[rowHID]:
                d.append(col_index)
                s.append(ukbHI.ID_number[rowHID])

filtereddata=Demo_all_PoP.iloc[:,d] #we filtered and extracted the columns that we wanted to look at them
    #and diagnose Super Healthy participants with them
title_org=filtereddata.columns
filtereddata.columns=s

sys.stdout.write(CYAN)
print("\n creating is finished")
sys.stdout.write(RESET)

sys.stdout.write(CYAN)
print("\n Filtering data based on cole criteria has been started")
sys.stdout.write(RESET)

[Hrow,Hcol]=filtereddata.shape


"""
after creating sub-DataFrame from UK BioBank Demographic_Phenotypic to only have Cole criteria's Field-ID,
We can now start extract Super Healthy participantc who also have MRI scans.

"""
hindex=0
sub=0
dd=[]

for sub in np.arange(rows1):
    rowHID=0
    zzz=filtereddata.iloc[sub,filtereddata.columns.get_loc(ukbHI.ID_number[rowHID])]
    if all(pd.isnull(zzz.iloc[:])):
        rowHID+=1
        zzz=filtereddata.iloc[sub,filtereddata.columns.get_loc(ukbHI.ID_number[rowHID])]
        if any(zzz==0) and not(all(pd.isnull(zzz))) and not(any(zzz==-3)) and not(any(zzz==1)) and not(any(zzz==-1)):
            rowHID+=1
            zzz=filtereddata.iloc[sub,filtereddata.columns.get_loc(ukbHI.ID_number[rowHID])]
            if any(zzz==0) and not(all(pd.isnull(zzz))) and not(any(zzz==-3)) and not(any(zzz==1)) and not(any(zzz==-1)):
                rowHID+=1
                zzz=filtereddata.iloc[sub,filtereddata.columns.get_loc(ukbHI.ID_number[rowHID])]
                if all(pd.isnull(zzz)):
                    rowHID+=1
                    zzz=filtereddata.iloc[sub,filtereddata.columns.get_loc(ukbHI.ID_number[rowHID])]
                    if (not(all(pd.isnull(zzz)))) and (not(any(zzz==-3))  and not(any(zzz==-1)) and not(any(zzz==4))) and (any(zzz==1) or any(zzz==2) or any(zzz==3)):
                        hindex+=1
                        dd.append(sub)


filtereddataSuperHealthyall=Demo_all_PoP.iloc[dd,:]
Demo_all_PoP.to_csv(Path_to_data.replace('CATs','Demographic_Phenotypic')+'Demographic_Phenotypic_MRI_PoP.csv')
filtereddataSuperHealthyall.to_csv(Path_to_data.replace('CATs','Demographic_Phenotypic')+'Demographic_Phenotypic__MRI_Super_Healthy.csv')

sys.stdout.write(CYAN)
print("\n Filtering data based on cole criteria has been Finished")
sys.stdout.write(RESET)
#%% Creating CAT csv files for Super Healthy participants
sys.stdout.write(CYAN)
print("\n Starting to create *.csv files for Super Healthy participants")
sys.stdout.write(RESET)

for file in os.listdir(Path_to_data.replace('CATs','CATs_Age')):
    print(file)
    Scheafer_Tian=pd.read_csv(Path_to_data.replace('CATs','CATs_Age')+file)
    Scheafer_Tian.drop(columns=['Unnamed: 0'],inplace=True)
    filtereddataSuperHealthyall.reset_index(drop=True,inplace=True)
    index_Super=[]
    print('index_Super:')
    print(index_Super)
    for i in np.arange(filtereddataSuperHealthyall.shape[0]):
        jj=Scheafer_Tian.subject_ID[Scheafer_Tian.subject_ID=='sub-'+str(filtereddataSuperHealthyall.eid[i])].index[0]
        index_Super.append(jj)
    CORTO_SUBCORTO_AGE_HEALTHY=Scheafer_Tian.iloc[index_Super,:]
    CORTO_SUBCORTO_AGE_HEALTHY.to_csv(Path_to_data.replace('CATs','CATs_Age_Healthy')+file.replace('Tian','Tian_Healthy'))
    print(Path_to_data.replace('CATs','CATs_Age_Healthy')+file.replace('Tian','Tian_Healthy'))

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