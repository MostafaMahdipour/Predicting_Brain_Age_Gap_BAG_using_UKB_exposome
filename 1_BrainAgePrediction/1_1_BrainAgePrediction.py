#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: 
    Mostafa Mahdipour              m.mahdipour@fz-juelich.de
"""
# %% 1 importing libraries

import sys
import os
from click import command
import numpy as np
import natsort

import pandas as pd # to use dataframe
import glob
import pickle

import nest_asyncio
nest_asyncio.apply()

import matplotlib.pyplot as plt # to make plots
import seaborn as sns # to make plots
import scipy # to calculate correlation

from sklearn.metrics import mean_absolute_error # to calculate MAE
from sklearn.metrics import mean_squared_error  # to calculate MSE
from sklearn.model_selection import KFold

from julearn import run_cross_validation
from julearn.utils import configure_logging
from julearn.pipeline import PipelineCreator
from julearn.model_selection import RepeatedContinuousStratifiedKFold
from sklearn.model_selection import train_test_split


import tqdm
tqdm.tqdm



# %% 2 Color print in terminal 
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

# %% 3 Starting statement
sys.stdout.write(RED)
# sys.stdout.write(BOLD)
print(
      "\n\n================================================="
      "\nprogram started\n"
      "=================================================\n\n"
      )
sys.stdout.write(CYAN)
print('Julearn info\n')
sys.stdout.write(RESET)
# To log information
configure_logging(level='INFO')
#%% 4 Pathes
My_current_path=os.path.abspath(os.getcwd())
# path to healthy subgroup
path_2_data_healthy = My_current_path + '[path to healthy data]'
# path to all subjects
path_2_data_all  = My_current_path + '[path to all data]'

# path to save results
Path_to_Save_Results = My_current_path + '[Path to save results]'

# check if the path exists
if not os.path.exists(Path_to_Save_Results):
    # If it doesn't exist, create it
    os.makedirs(Path_to_Save_Results)
    sys.stdout.write(GREEN)
    print(f"Directory '{Path_to_Save_Results}' created successfully.")
    sys.stdout.write(RESET)
else:
    sys.stdout.write(RED)
    print(f"Directory '{Path_to_Save_Results}' already exists.")
    sys.stdout.write(RESET)
#%% 5 loading Data
'''
As mentioned in the README file, we have 5 different granularitys:
- Schaefer_17Network_200_UKB + Tian S2
- Schaefer_17Network_400_UKB + Tian S4
- Schaefer_17Network_600_UKB + Tian S4
- Schaefer_17Network_800_UKB + Tian S4
- Schaefer_17Network_1000_UKB + Tian S4

Therefore, we have first check the files in the directory and then load the them.
'''
# ls function : This is a function that can do ls in a directory
def listdir_nohidden(path):
    return glob.glob(os.path.join(path, '*')) # to make a list of our folders' contents
list_of_input_files_healthy=listdir_nohidden(path_2_data_healthy)
list_of_input_files_all=listdir_nohidden(path_2_data_all)
print(natsort.natsorted(list_of_input_files_healthy))
'''
as you can see in the print output we have 5 different datasets.
'''
# Here we load the healthy and all data for the biggest granularity (1000) and the Tian dataset:

# loading the healthy data : natsort.natsorted(list_of_input_files_healthy)[4] ==> 'Age_Schaefer_17Network_10000_UKB_Tian_Healthy.csv'
Healthy_Data=pd.read_csv(natsort.natsorted(list_of_input_files_healthy)[4],index_col=0).reset_index(drop=True)
# loading the healthy data : natsort.natsorted(list_of_input_files_all)[4] ==> 'Age_Schaefer_17Network_10000_UKB_Tian.csv'
All_Data=pd.read_csv(natsort.natsorted(list_of_input_files_all)[4],index_col=0).reset_index(drop=True)
PoP_Data=All_Data[~All_Data.SubjectID.isin(Healthy_Data.SubjectID)] # Population = All - healthy

#%% 6 parentheses
All_Data.columns = All_Data.columns.str.replace(')', '')
All_Data.columns = All_Data.columns.str.replace('(', '')
All_Data.columns = All_Data.columns.str.replace(',', '')
All_Data.columns = All_Data.columns.str.replace('/', '')
#%% 7 Features and data types
Data2Model= Healthy_Data.drop(columns=[ 'NCR',
                                    'ICR',
                                    'IQR',
                                    'TIV',
                                    'GM',
                                    'WM',
                                    'CSF',
                                    'WMH',
                                    'TSA',])

y = 'Age' # target
X_list = list(set(Data2Model.columns) - set(['Unnamed: 0', 
                                          'SubjectID',
                                          'Session',
                                          'Age',
                                          'Sex'])) # features & confounds

X_types = {'continuous': X_list}
#%% 8 Pipeline
rand_seed = 94
creator = PipelineCreator(problem_type="regression", 
                          apply_to="continuous")
creator.add("zscore")
# here we add the model. Models We have used:
# 'linreg' ==> 'linear_regression', there is no hyperparameter for it to be tuned 
# 'ridge' ==> 'ridge', we have tuned alpha as a hyperparameter (see below) 
# 'svr' ==> 'linear_svr', we have tuned C, kernel, and gamma as a hyperparameters (see Julearn documentation)
# 'rf' ==> 'random_forest', we have tuned n_estimators, min_samples_split, min_samples_leaf,and max_depth hyperparameters (see Julearn documentation)
creator.add("ridge", 
            alpha=[0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 20, 50, 80, 100, 150, 200, 300],
            fit_intercept=[True, False])
sys.stdout.write(RED)
sys.stdout.write(BOLD)
sys.stdout.write(CGREENBG)
print(creator)
sys.stdout.write(RESET)

#%% 9 Nested CV and scores
train_df, test_df = train_test_split(Data2Model, test_size=0.2, random_state=rand_seed)

cv_splitter = RepeatedContinuousStratifiedKFold(5, 
                                                    n_splits=10, n_repeats=5, 
                                                    random_state=rand_seed, 
                                                    method='quantile')

scoring = ['neg_mean_absolute_error', 'neg_mean_squared_error',
                                                    'neg_root_mean_squared_error',
                                                    'r2','r2_corr',
                                                    'r_corr']

search_params = {
    "kind": "grid",
    "cv": KFold(n_splits=10, shuffle = True, random_state=rand_seed),
    "verbose" : 1
}
#%% 10 run_cross_validation : 
scores, model, inspector = run_cross_validation(
    X=X_list,
    y=y,
    data=train_df,
    X_types=X_types,
    model=creator,
    return_train_score=True,
    return_estimator="all",
    return_inspector=True,
    seed=rand_seed, 
    cv=cv_splitter, 
    scoring=scoring, 
    search_params=search_params,
    n_jobs=4
)

sys.stdout.write(BLUE)
print('best para', model.best_params_)
sys.stdout.write(RESET)

sys.stdout.write(RED)

print(
      "\n\n================================================="
      "\nModel Trained\n"
      "=================================================\n\n"
      )
sys.stdout.write(RESET)
# %% 11 saving
# Save the model using pickle
with open(Path_to_Save_Results+'/Model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Save the inspector using pickle
with open(Path_to_Save_Results+'/inspector.pkl', 'wb') as file:
    pickle.dump(inspector, file)

# Save scores using pickle
with open(Path_to_Save_Results+'/scores.pkl', 'wb') as file:
    pickle.dump(scores, file)

# %%
