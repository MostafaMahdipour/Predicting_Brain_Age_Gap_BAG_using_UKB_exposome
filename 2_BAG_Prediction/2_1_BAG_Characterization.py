#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 06 2023

@author: 
    Mostafa Mahdipour              m.mahdipour@fz-juelich.de
"""


"""
# Brian Age Gap (BAG) prediction
In this code, we try to change the features into continuous and categorical
then using uptona for hyperparameter tuning.
"""
# %% 1 importing libraries

import sys
import os
import ast # convert string model_params to dictionary
from click import command
import numpy as np
# import wget
import pandas as pd # to use dataframe
import glob
import pickle
# import psutil

from pprint import pprint  # To print in a pretty way
# import missingno as msno
import nest_asyncio
nest_asyncio.apply()


import matplotlib.pyplot as plt # to make plots
import seaborn as sns # to make plots
import scipy # to calculate correlation

import optuna
from optuna.distributions import IntDistribution
from optuna.distributions import FloatDistribution

from sklearn.metrics import mean_absolute_error # to calculate MAE
from sklearn.metrics import mean_squared_error  # to calculate MSE
from sklearn import linear_model
from sklearn.model_selection import KFold

# from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold

from julearn import run_cross_validation
from julearn.utils import configure_logging
from julearn.pipeline import PipelineCreator
from julearn.model_selection import RepeatedContinuousStratifiedKFold
# from julearn.model_selection import ContinuousStratifiedKFold


import tqdm
tqdm.tqdm
import shap

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


# Checking which conda env are we using
sys.stdout.write(GREEN)
# sys.stdout.write(BOLD)
print(
      "\nvirtual conda env:\n"
      )
print(sys.executable)
sys.stdout.write(RESET)
#%% 4 Pathes
My_current_path=os.path.abspath(os.getcwd())
path_2_data_NEW = My_current_path + 'Path/to/exposome/Subgroups' # path to the exposome subgroups
# for exmple:
# path_2_data_NEW = My_current_path + '/../../../../' + '1_DATA/4_BAG_Prediction/9_DATA_4_BAG_GMV_SHAP'
result_folder_Name= 'name of the folder' # name of the folder to save the results
# for exmple:
# result_folder_Name= '3_BAG_CR_RF' # 3 ==> subset # 3 or main subset, BAG ==> brain age gap,  CR ==> with Confound Removal, RF ==> Random Forest
Path_to_Save_Results = My_current_path + '/path/to/results/'+ result_folder_Name

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
#%% 5 creating optuna study
'''
If we run the code for the first time, we will get an error because there is no study to delete
We can ignore the error and create the study with the "try & except" approach.
'''


study_name = result_folder_Name
storage_name = "sqlite:///"+result_folder_Name+".db"

try:
    optuna.delete_study(study_name=study_name, storage=storage_name)
    print(f"Study '{study_name}' deleted successfully.")
except KeyError:
    print(f"Warning: The study '{study_name}' does not exist. Continuing with the next steps...")

# optuna.delete_study(study_name="12_6_226_BAG_all_CR_RF", storage="sqlite:///rf-test.db")
study = optuna.create_study(
    direction = "maximize",
    storage = storage_name,
    study_name = result_folder_Name,
    load_if_exists=True,
) 
sys.stdout.write(CGREENBG)
sys.stdout.write(UNDERLINE)
sys.stdout.write(BLUE)
# sys.stdout.write(BOLD)
print("\n\n=================================================")
print(f"The study '{result_folder_Name}' has been created successfully.")
print("=================================================\n\n")
sys.stdout.write(RESET)
#%% 6 loading Data

# ls function : This is a function that can do ls in a directory
def listdir_nohidden(path):
    return glob.glob(os.path.join(path, '*')) # to make a list of our folders' contents
list_of_input_files=listdir_nohidden(path_2_data_NEW)

print(np.sort(list_of_input_files))
'''
As mentioned in the README file, we have 3 different subgroups of exposome data:
- subgroup1.csv
- subgroup2.csv
- subgroup3.csv
And a list of categorical features:
- listofcols2Cat.npy

Therefore, we have first check the files in the directory and then load the them.
'''
# Here we load the subgroup3 and listofcols2Cat:
# (we can change the index to load other datasets)

# loading the data : np.sort(list_of_input_files)[2] ==> 'subgroup3.csv'
All_Data=pd.read_csv(np.sort(list_of_input_files)[2],index_col=0).reset_index(drop=True)
# loading the Categorical data labels : np.sort(list_of_input_files)[-1] ==> 'listofcols2Cat.npy'
loaded_list = np.load(np.sort(list_of_input_files)[-1], allow_pickle=True)

#%% 7 parentheses
All_Data.columns = All_Data.columns.str.replace(')', '')
All_Data.columns = All_Data.columns.str.replace('(', '')
All_Data.columns = All_Data.columns.str.replace(',', '')
All_Data.columns = All_Data.columns.str.replace('/', '')

# Perform string replacements on each element in the list
listofcols2Category = [col.replace(')', '').replace('(', '').replace(',', '').replace('/', '') for col in loaded_list]
#%% 8 Features and data types
"""
defining Features and data types
# X_list
    - confounds
    - features
        -- categorical
        -- continuous
"""
Data2Model= All_Data.drop(columns=['TIV','GM', 
                                  'BAG', 'Session', 
                                  'Predicted_Age', 'Corrected_Predicted_Age'])

y = 'Corrected_BAG' # target
X_list = list(set(Data2Model.columns) - set(['SubjectID',
                                          'Corrected_BAG'])) # features & confounds

Our_Confounds = ['Age','Age2', 'Height-2.0', 'Sex',
                'Volumetric_scaling_from_T1_head_image_to_standard_space-2.0']
Our_Features = list(set(X_list) - set(Our_Confounds))

common_elements = np.intersect1d(Our_Features, listofcols2Category) # the categorical features in our dataset

ContinuousVars = list(set(Our_Features) - set(common_elements))

# define features and confounds
# {'categorical' : listofcols2Category, 'continuous' : ContinuousVars})
X_types = {'continuous': list(ContinuousVars), 'categorical' : list(common_elements), 'confounds': list(Our_Confounds)}

#%% 9 Pipeline
rand_seed = 94
creator = PipelineCreator(problem_type="regression", apply_to=["continuous","categorical"])
creator.add("zscore", apply_to="continuous")
creator.add("confound_removal", confounds="confounds", apply_to="continuous")

# here we add the model. Models We have used:
# 'linreg' ==> 'linear_regression', there is no hyperparameter for it to be tuned 
# 'ridge' ==> 'ridge', we have tuned alpha as a hyperparameter (see below) 
# 'svr' ==> 'linear_svr', we have tuned C, kernel, and gamma as a hyperparameters (see Julearn documentation)
# 'rf' ==> 'random_forest', we have tuned n_estimators, min_samples_split, min_samples_leaf,and max_depth hyperparameters (see Julearn documentation)

# the example of RandomForest hyperparameters:
creator.add("rf",
            n_estimators=(200,1000,"uniform"),
            min_samples_split=(1,10, "uniform"),
            min_samples_leaf=(1,40, "uniform"),
            max_depth = (20,180, "uniform")
)


sys.stdout.write(RED)
sys.stdout.write(BOLD)
sys.stdout.write(CGREENBG)
print(creator)
sys.stdout.write(RESET)
#%% 10 Nested CV and scores

cv_splitter = RepeatedContinuousStratifiedKFold(3, 
                                                    n_splits=5, n_repeats=5, 
                                                    random_state=rand_seed, 
                                                    method='quantile')

scoring = ['neg_mean_absolute_error', 'neg_mean_squared_error',
                                                    'neg_root_mean_squared_error',
                                                    'r2','r2_corr',
                                                    'r_corr']

search_params = {
    "kind": "optuna",
    "n_trials": 100,
    "study": study,
    "cv": KFold(n_splits=5, shuffle = True, random_state=rand_seed),
    "verbose" : 1
}
#%% 11 run_cross_validation : 
scores, model, inspector = run_cross_validation(
    X=X_list,
    y=y,
    data=Data2Model,
    X_types=X_types,
    model=creator,
    return_train_score=True,
    return_estimator="all",
    return_inspector=True,
    seed=rand_seed, 
    cv=cv_splitter, 
    scoring=scoring, 
    search_params=search_params,
    n_jobs=4, # as Fede sugested n_jobs should be set to 4
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
# %% 12 saving
# Save the model using pickle
with open(Path_to_Save_Results+'/Model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Save the inspector using pickle
with open(Path_to_Save_Results+'/inspector.pkl', 'wb') as file:
    pickle.dump(inspector, file)

# Save scores using pickle
with open(Path_to_Save_Results+'/scores.pkl', 'wb') as file:
    pickle.dump(scores, file)


# %% 13 extra plots plot


if not os.path.exists(Path_to_Save_Results+'/Outerfold_extra'):
    # If it doesn't exist, create it
    os.makedirs(Path_to_Save_Results+'/Outerfold_extra')
    sys.stdout.write(GREEN)
    print(f"Directory '{Path_to_Save_Results+'/Outerfold_extra'}' created successfully.")
    sys.stdout.write(RESET)
else:
    sys.stdout.write(RED)
    print(f"Directory '{Path_to_Save_Results+'/Outerfold_extra'}' already exists.")
    sys.stdout.write(RESET)



for i, (train_index, test_index) in enumerate(cv_splitter.split(Data2Model.loc[:,X_list], Data2Model.loc[:,y])):
    # sys.stdout.write(RED)
    # print("\n\n=================================================\n\n")
    # print(f"Fold {i}:")
    # sys.stdout.write(RESET)

    y_pred = scores['estimator'][i].predict(Data2Model.iloc[test_index,:].loc[:,X_list]).ravel()
    y_true=Data2Model.iloc[test_index,:].loc[:,y]
    mae=mean_absolute_error(y_true, y_pred)
    mse=mean_squared_error(y_true, y_pred)
    corr, p = scipy.stats.pearsonr(y_pred, y_true)
    # sys.stdout.write(CYAN)
    # print('Test MSE: ', mse ,'Test MAE: ', mae, 'Test Corr: ', corr)
    # sys.stdout.write(RESET)
    results_df_test = pd.DataFrame({'True BAG': y_true, 'Predicted BAG': y_pred})
    # Generate scatter plot
    fig100=plt.figure('scatter predicted BAG vs True BAG',figsize=(16,16))
    ax11=sns.jointplot(data=results_df_test, x="True BAG", y="Predicted BAG", 
                    kind="kde",fill=True, joint_kws={'alpha': 0.5} ,
                        color='Darkorange')  # peru
    ax11.ax_joint.set_xlim(-20,20)
    ax11.ax_joint.set_ylim(-20,20)

    ax11.ax_joint.plot([-20,20], [-20,20], 'r-', linewidth = 2)
    ax11.ax_joint.tick_params(axis='both', labelsize=20)
    sns.regplot(data=results_df_test, x='True BAG', y='Predicted BAG',
        scatter=False, color='black', line_kws={'linestyle':'--'}, ax=ax11.ax_joint)

    #calculate slope and intercept of regression equation
    slope_1, intercept_1, r_1, p_1, sterr_1 = scipy.stats.linregress( x=ax11.ax_joint.get_lines()[1].get_xdata(),
                                                                      y=ax11.ax_joint.get_lines()[1].get_ydata())


    # Add regression equation to the joint plot
    ax11.ax_joint.text(8, -10, 'y = ' + str(round(intercept_1, 3)) + ' + ' + str(round(slope_1, 3)) + 'x')


    plt.grid()
    # plt.show()
    # Save 
    results_df_test.to_csv(Path_to_Save_Results+'/Outerfold_extra' + '/outer_CV_fold'+str(i)+'.csv')

    # # del results_df_test
    plt.savefig(Path_to_Save_Results+'/Outerfold_extra' + '/FOLD'+str(i)+'.png')

    plt.close(fig100)
    plt.close('all')
    del results_df_test
    del intercept_1
    del slope_1
    del y_pred
    del y_true
    del fig100
    del ax11

# save scores
scores.to_csv(Path_to_Save_Results+'/Outerfold_extra'+'/scores.csv')

# %% 14 ending statement

sys.stdout.write(RED)
print(
      "\n\n================================================="
      "\nProgramm ended\n"
      "=================================================\n\n"
      )
sys.stdout.write(RESET)
# %%
