#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

There are two sub-folders in "../3_scratch/0_for_merge" directory:
1-"Somayeh": which contains the Schaefer and Tian Atlases with different parcellations that Somayeh has calculated for us.
 If we can have these *.csv files on the datalad dataset, we can clone them from there.

2-"Felix": Which contains the Schaefer Atlas with 600 parcellations. We want this file only for extracting
 {NCR', 'ICR', 'IQR', 'TIV', 'GM', 'WM', 'CSF', 'WMH', 'TSA'} columns and add them to our CAT results. 
  In case you want to use the last version of this file, you should first create a suitable 
  anaconda/miniconda environment and install the datalad and its other dependencies on your Juseless account. 
  Then, you can run the below commands to download this file:
      
      datalad clone -d . 'ria+http://ukb.ds.inm7.de#~cat_m0wp1' inputs/cat12.7
      datalad get inputs/cat12.7/stats/cat_rois_Schaefer2018_600Parcels_17Networks_order.csv




@author: mmahdipour
"""
#%%%
import sys
import numpy as np
import pandas as pd
import glob
import os
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
#%%
def listdir_nohidden(path):
    return glob.glob(os.path.join(path, '*')) # to make a list of our folders' contents


list_files_Somayeh=listdir_nohidden('../3_scratch/0_for_merge/Somayeh')
list_files_Felix=listdir_nohidden('../3_scratch/0_for_merge/Felix')
Felix_=pd.read_csv(list_files_Felix[0])
Felix_=Felix_.sort_values(by=['SubjectID','Session']) #sort base on these two columns
Felix_=Felix_.drop_duplicates(subset='SubjectID',keep='first') # delete participants who have two sessions(keep first session)

Felix_.reset_index(drop=True,inplace=True) # with removing the first row the index 0 will miss, 
    # with np.reset_index() we can reset the indexing. note: the drop option removes the original index column
#%%
for i in list_files_Somayeh:
    print(i)
    temp=pd.read_csv(i)
    temp.columns=temp.iloc[0] # to replace the first row in the columns title
    temp=temp.drop(labels=0,axis=0,inplace=False) # to remove the first row as it is not integer
    # temp.reset_index(drop=True,inplace=True) # with removing the first row the index 0 will miss, 
    # with np.reset_index() we can reset the indexing. note: the drop option removes the original index column
    temp=temp.sort_values(by=['subject_ID','session'])
    temp=temp.drop_duplicates(subset='subject_ID',keep='first')
    #inserting the name of missed columns in the second dataframe. Each time we insert a new column in postion 2
    # other columns go one step to the left side
    temp.insert(2,'TSA',np.nan)
    temp.insert(2,'WMH',np.nan)
    temp.insert(2,'CSF',np.nan)
    temp.insert(2,'WM',np.nan)
    temp.insert(2,'GM',np.nan)
    temp.insert(2,'TIV',np.nan)
    temp.insert(2,'IQR',np.nan)
    temp.insert(2,'ICR',np.nan)
    temp.insert(2,'NCR',np.nan)
    for j in np.arange(Felix_.shape[0]):
        #comment fot below command:
        #find the index of corresponding person from first dataframe (Felix) in the second (Somayeh)
        jj=temp.subject_ID[temp.subject_ID==Felix_.SubjectID[Felix_.index[j]]].index[0]
        if Felix_.Session[j] == temp.session[jj]: #extra check for sessions (Time instances)
            temp.TSA[jj]=Felix_.TSA[j]
            temp.WMH[jj]=Felix_.WMH[j]
            temp.CSF[jj]=Felix_.CSF[j]
            temp.WM[jj]=Felix_.WM[j]
            temp.GM[jj]=Felix_.GM[j]
            temp.TIV[jj]=Felix_.TIV[j]
            temp.IQR[jj]=Felix_.IQR[j]
            temp.ICR[jj]=Felix_.ICR[j]
            temp.NCR[jj]=Felix_.NCR[j]
    #delete rows with NaN value in their new columns (Somayeh's file has two extra participants)
    #but we don't have their GM values so unfortunatly we have to ignore them
    temp.dropna(inplace=True)
    temp.reset_index(drop=True,inplace=True)
    test1=i[29:]
    if test1.startswith('Tian'): # remove these columns from Tian, because we have already put
    #them in the Schaefer Atlas
        temp.drop(columns=['TSA','WMH','CSF','WM','GM','TIV','IQR','ICR','NCR'],inplace=True)
    temp.to_csv(i.replace('3_scratch/0_for_merge/Somayeh','1_DATA/CATs'))
# %% Ending statement
sys.stdout.write(RED)

print(
      "\n\n================================================="
      "\nprogram ended\n"
      "=================================================\n\n"
      )
sys.stdout.write(RESET)
