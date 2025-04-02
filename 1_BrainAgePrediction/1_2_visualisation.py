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
path_2_data_Source = My_current_path + '/../../../' + '1_DATA/'
path_2_data_healthy = path_2_data_Source + 'CATs_Age_Healthy'
path_2_data_all  = path_2_data_Source + 'CATs_Age'

result_folder_Name= '2_2_1000_BA_Ridge'
Path_to_Save_Results = My_current_path + '/../../../' + '4_Results/2_Brain_Age_Prediction/'+ result_folder_Name

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

# ls function : This is a function that can do ls in a directory
def listdir_nohidden(path):
    return glob.glob(os.path.join(path, '*')) # to make a list of our folders' contents
list_of_input_files_healthy=listdir_nohidden(path_2_data_healthy)
list_of_input_files_all=listdir_nohidden(path_2_data_all)
print(natsort.natsorted(list_of_input_files_healthy))
'''
as you can see in the print output we have 5 different datasets.
'''
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
# define categorical and continuous
# listofcols2Category = ['Sex']
# ContinuousVars = list(set(X_list)- set(listofcols2Category))
# X_types = {'categorical': listofcols2Category, 'continuous': ContinuousVars}
X_types = {'continuous': X_list}
#%% 8 Pipeline
rand_seed = 94
creator = PipelineCreator(problem_type="regression", 
                          apply_to="continuous")
creator.add("zscore")
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
    # "cv": 3, 
    "cv": KFold(n_splits=10, shuffle = True, random_state=rand_seed),
    "verbose" : 1
}
 #%% 10 run_cross_validation : 
# scores, model, inspector = run_cross_validation(
#     X=X_list,
#     y=y,
#     data=train_df,
#     X_types=X_types,
#     model=creator,
#     return_train_score=True,
#     return_estimator="all",
#     return_inspector=True,
#     seed=rand_seed, 
#     cv=cv_splitter, 
#     scoring=scoring, 
#     search_params=search_params,
#     n_jobs=4
# )

# sys.stdout.write(BLUE)
# print('best para', model.best_params_)
# sys.stdout.write(RESET)

# sys.stdout.write(RED)

# print(
#       "\n\n================================================="
#       "\nModel Trained\n"
#       "=================================================\n\n"
#       )
# sys.stdout.write(RESET)
# %% 11 saving

# Load the model
with open(Path_to_Save_Results+'/Model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the inspector
with open('SVR_Test_inspector.pkl', 'rb') as file:
    inspector = pickle.load(file)

# Load the scores
with open('SVR_Test_scores.pkl', 'rb') as file:
    scores = pickle.load(file)


if not os.path.exists(Path_to_Save_Results+'/extra_results'):
    # If it doesn't exist, create it
    os.makedirs(Path_to_Save_Results+'/extra_results')
    sys.stdout.write(GREEN)
    print(f"Directory '{Path_to_Save_Results+'/extra_results'}' created successfully.")
    sys.stdout.write(RESET)
else:
    sys.stdout.write(RED)
    print(f"Directory '{Path_to_Save_Results+'/extra_results'}' already exists.")
    sys.stdout.write(RESET)

# %% 12 plot and calculate the slope and intercept from train set:
#predicting the whole tarin set
y_pred_train = model.best_estimator_.predict(train_df[X_list])
# y_pred_train = model.predict(train_df[X_list])
y_true_train = train_df[y]
results_df_train = train_df[['SubjectID', 'Session', 'Sex']].copy()

results_df_train['True Age Train'] = y_true_train
results_df_train['Predicted Age Train'] = y_pred_train

fig100=plt.figure('scatter predicted Age vs True Age',figsize=(16,16))
ax11=sns.jointplot(data=results_df_train, x="True Age Train", y="Predicted Age Train", 
                kind="kde",fill=True, joint_kws={'alpha': 0.5} ,
                    color='#00A693')  # peru
x1=35
y1=35
x2=100
y2=100

ax11.ax_joint.set_xlim(x1,x2)
ax11.ax_joint.set_ylim(y1,y2)

# ax11.ax_joint.plot([x1,x2], [y1,y2], 'r-', linewidth = 2)
ax11.ax_joint.plot([x1, x2], [y1, y2], color='#A21441', linestyle='-', linewidth=2)

ax11.ax_joint.tick_params(axis='both', labelsize=20)
sns.regplot(data=results_df_train, x='True Age Train', y='Predicted Age Train',
    scatter=False, color='black', line_kws={'linestyle':'--'}, ax=ax11.ax_joint)

#calculate slope and intercept of regression equation
slope_train, intercept_train, r_train, p_train, sterr_train = scipy.stats.linregress( x=ax11.ax_joint.get_lines()[1].get_xdata(),
                                                                y=ax11.ax_joint.get_lines()[1].get_ydata())


# Add regression equation to the joint plot
ax11.ax_joint.text(50, 80, 'y = ' + str(round(intercept_train, 3)) + ' + ' + str(round(slope_train, 3)) + 'x')
# plt.title('Brain Age prediction on healthy set (train set)')
plt.grid()
# plt.show()
plt.savefig(Path_to_Save_Results+'/extra_results' + '/1_Healthy_train_set.png')
plt.close(fig100)
plt.close('all')
del fig100
del ax11

MAE_train = mean_absolute_error(results_df_train['True Age Train'],results_df_train['Predicted Age Train'])
MSE_train = mean_squared_error(results_df_train['True Age Train'],results_df_train['Predicted Age Train'])
corr_train, p_val_train = scipy.stats.pearsonr(results_df_train['True Age Train'],results_df_train['Predicted Age Train'])
# %% test set : plot brain age + BAG + Bias correction  + plot bais corrected brain age +  bais corrected BAG

"""plot brain age"""
y_true_test = test_df[y]
y_pred_test = model.best_estimator_.predict(test_df[X_list])
results_df_test = test_df[['SubjectID', 'Session', 'Sex']].copy()
results_df_test['True Age Test'] = y_true_test
results_df_test['Predicted Age Test'] = y_pred_test

fig100=plt.figure('scatter predicted Age vs chronological Age test set',figsize=(16,16))
ax11=sns.jointplot(data=results_df_test, x="True Age Test", y="Predicted Age Test", 
                kind="kde",fill=True, joint_kws={'alpha': 0.5} ,
                    color='#1C39BB')  # peru
x1=35
y1=35
x2=100
y2=100

ax11.ax_joint.set_xlim(x1,x2)
ax11.ax_joint.set_ylim(y1,y2)

# ax11.ax_joint.plot([x1,x2], [y1,y2], 'r-', linewidth = 2)
ax11.ax_joint.plot([x1, x2], [y1, y2], color='#A21441', linestyle='-', linewidth=2)

ax11.ax_joint.tick_params(axis='both', labelsize=20)
sns.regplot(data=results_df_test, x='True Age Test', y='Predicted Age Test',
    scatter=False, color='black', line_kws={'linestyle':'--'}, ax=ax11.ax_joint)

#calculate slope and intercept of regression equation
slope_test, intercept_test, r_test, p_test, sterr_test = scipy.stats.linregress( x=ax11.ax_joint.get_lines()[1].get_xdata(),
                                                                y=ax11.ax_joint.get_lines()[1].get_ydata())


# Add regression equation to the joint plot
ax11.ax_joint.text(50, 80, 'y = ' + str(round(intercept_test, 3)) + ' + ' + str(round(slope_test, 3)) + 'x')
# plt.title('Brain Age prediction on healthy set (test set)')
plt.grid()
# plt.show()
plt.savefig(Path_to_Save_Results+'/extra_results' + '/2_Healthy_test_set_before_Bias_correction.png')
plt.close(fig100)
plt.close('all')
del fig100
del ax11

MAE_test = mean_absolute_error(results_df_test['True Age Test'],results_df_test['Predicted Age Test'])
MSE_test = mean_squared_error(results_df_test['True Age Test'],results_df_test['Predicted Age Test'])
corr_test, p_val_test = scipy.stats.pearsonr(results_df_test['True Age Test'],results_df_test['Predicted Age Test'])

"""BAG"""
results_df_test['BAG Test'] = results_df_test['Predicted Age Test'] - results_df_test['True Age Test']

fig100=plt.figure('scatter chronological Age vs BAG test set',figsize=(16,16))
ax11=sns.jointplot(data=results_df_test, x="True Age Test", y="BAG Test", 
                kind="kde",fill=True, joint_kws={'alpha': 0.5} ,
                    color='#1C39BB')  # peru
x1=35
y1=-20
x2=100
y2=20

ax11.ax_joint.set_xlim(x1,x2)
ax11.ax_joint.set_ylim(y1,y2)

ax11.ax_joint.plot([x1, x2], [0, 0], color='#A21441', linestyle='-', linewidth=2)

ax11.ax_joint.tick_params(axis='both', labelsize=20)
sns.regplot(data=results_df_test, x='True Age Test', y='BAG Test',
    scatter=False, color='black', line_kws={'linestyle':'--'}, ax=ax11.ax_joint)

#calculate slope and intercept of regression equation
slope_BAG_test, intercept_BAG_test, r_BAG_test, p_BAG_test, sterr_BAG_test = scipy.stats.linregress( x=ax11.ax_joint.get_lines()[1].get_xdata(),
                                                                y=ax11.ax_joint.get_lines()[1].get_ydata())


# Add regression equation to the joint plot
ax11.ax_joint.text(70, 6, 'y = ' + str(round(intercept_BAG_test, 3)) + ' + ' + str(round(slope_BAG_test, 3)) + 'x')

plt.grid()
# plt.show()
plt.savefig(Path_to_Save_Results+'/extra_results' + '/3_Healthy_test_set_BAG_before_Bias_correction.png')
plt.close(fig100)
plt.close('all')
del fig100
del ax11
"""Bias correction"""

results_df_test['Corrected predicted Age Test'] = (results_df_test['Predicted Age Test']-intercept_train)/slope_train

"""plot bais corrected brain age"""
fig100=plt.figure('scatter corrected predicted Age vs chronological Age test set',figsize=(16,16))
ax11=sns.jointplot(data=results_df_test, x="True Age Test", y="Corrected predicted Age Test", 
                kind="kde",fill=True, joint_kws={'alpha': 0.5} ,
                    color='#1C39BB')  # peru
x1=35
y1=35
x2=100
y2=100

ax11.ax_joint.set_xlim(x1,x2)
ax11.ax_joint.set_ylim(y1,y2)

# ax11.ax_joint.plot([x1,x2], [y1,y2], 'r-', linewidth = 2)
ax11.ax_joint.plot([x1, x2], [y1, y2], color='#A21441', linestyle='-', linewidth=2)

ax11.ax_joint.tick_params(axis='both', labelsize=20)
sns.regplot(data=results_df_test, x='True Age Test', y='Corrected predicted Age Test',
    scatter=False, color='black', line_kws={'linestyle':'--'}, ax=ax11.ax_joint)

#calculate slope and intercept of regression equation
slope_c_test, intercept_c_test, r_c_test, p_c_test, sterr_c_test = scipy.stats.linregress( x=ax11.ax_joint.get_lines()[1].get_xdata(),
                                                                y=ax11.ax_joint.get_lines()[1].get_ydata())


# Add regression equation to the joint plot
ax11.ax_joint.text(50, 80, 'y = ' + str(round(intercept_c_test, 3)) + ' + ' + str(round(slope_c_test, 3)) + 'x')
# plt.title('Brain Age prediction on healthy set (test set)')
plt.grid()
# plt.show()
plt.savefig(Path_to_Save_Results+'/extra_results' + '/4_Healthy_test_set_after_Bias_correction.png')
plt.close(fig100)
plt.close('all')
del fig100
del ax11

MAE_c_test = mean_absolute_error(results_df_test['True Age Test'],results_df_test['Corrected predicted Age Test'])
MSE_c_test = mean_squared_error(results_df_test['True Age Test'],results_df_test['Corrected predicted Age Test'])
corr_c_test, p_val_c_test = scipy.stats.pearsonr(results_df_test['True Age Test'],results_df_test['Corrected predicted Age Test'])

"""bais corrected BAG"""
results_df_test['Corrected BAG Test'] = results_df_test['Corrected predicted Age Test'] - results_df_test['True Age Test']
fig100=plt.figure('scatter chronological Age vs corrected BAG test set',figsize=(16,16))
ax11=sns.jointplot(data=results_df_test, x="True Age Test", y="Corrected BAG Test", 
                kind="kde",fill=True, joint_kws={'alpha': 0.5} ,
                    color='#1C39BB')  # peru
x1=35
y1=-20
x2=100
y2=20

ax11.ax_joint.set_xlim(x1,x2)
ax11.ax_joint.set_ylim(y1,y2)

ax11.ax_joint.plot([x1, x2], [0, 0], color='#A21441', linestyle='-', linewidth=2)

ax11.ax_joint.tick_params(axis='both', labelsize=20)
sns.regplot(data=results_df_test, x='True Age Test', y='Corrected BAG Test',
    scatter=False, color='black', line_kws={'linestyle':'--'}, ax=ax11.ax_joint)

#calculate slope and intercept of regression equation
slope_cBAG_test, intercept_cBAG_test, r_cBAG_test, p_cBAG_test, sterr_cBAG_test = scipy.stats.linregress( x=ax11.ax_joint.get_lines()[1].get_xdata(),
                                                                y=ax11.ax_joint.get_lines()[1].get_ydata())


# Add regression equation to the joint plot
ax11.ax_joint.text(70, 6, 'y = ' + str(round(intercept_cBAG_test, 3)) + ' + ' + str(round(slope_cBAG_test, 3)) + 'x')

plt.grid()
# plt.show()
plt.savefig(Path_to_Save_Results+'/extra_results' + '/5_Healthy_test_set_BAG_after_Bias_correction.png')
plt.close(fig100)
plt.close('all')
del fig100
del ax11

# %% PoP set : plot brain age + BAG + Bias correction  + plot bais corrected brain age +  bais corrected BAG

"""plot brain age"""
y_true_PoP = PoP_Data[y]
y_pred_PoP = model.best_estimator_.predict(PoP_Data[X_list])
results_df_PoP = PoP_Data[['SubjectID', 'Session', 'Sex']].copy()
results_df_PoP['True Age PoP'] = y_true_PoP
results_df_PoP['Predicted Age PoP'] = y_pred_PoP

fig100=plt.figure('scatter predicted Age vs chronological Age PoP set',figsize=(16,16))
ax11=sns.jointplot(data=results_df_PoP, x="True Age PoP", y="Predicted Age PoP", 
                kind="kde",fill=True, joint_kws={'alpha': 0.5} ,
                    color='#D45814')  # peru
x1=35
y1=35
x2=100
y2=100

ax11.ax_joint.set_xlim(x1,x2)
ax11.ax_joint.set_ylim(y1,y2)

# ax11.ax_joint.plot([x1,x2], [y1,y2], 'r-', linewidth = 2)
ax11.ax_joint.plot([x1, x2], [y1, y2], color='#A21441', linestyle='-', linewidth=2)

ax11.ax_joint.tick_params(axis='both', labelsize=20)
sns.regplot(data=results_df_PoP, x='True Age PoP', y='Predicted Age PoP',
    scatter=False, color='black', line_kws={'linestyle':'--'}, ax=ax11.ax_joint)

#calculate slope and intercept of regression equation
slope_PoP, intercept_PoP, r_PoP, p_PoP, sterr_PoP = scipy.stats.linregress( x=ax11.ax_joint.get_lines()[1].get_xdata(),
                                                                y=ax11.ax_joint.get_lines()[1].get_ydata())


# Add regression equation to the joint plot
ax11.ax_joint.text(50, 80, 'y = ' + str(round(intercept_PoP, 3)) + ' + ' + str(round(slope_PoP, 3)) + 'x')
plt.grid()
# plt.show()
plt.savefig(Path_to_Save_Results+'/extra_results' + '/6_PoP_set_before_Bias_correction.png')
plt.close(fig100)
plt.close('all')
del fig100
del ax11

MAE_PoP = mean_absolute_error(results_df_PoP['True Age PoP'],results_df_PoP['Predicted Age PoP'])
MSE_PoP = mean_squared_error(results_df_PoP['True Age PoP'],results_df_PoP['Predicted Age PoP'])
corr_PoP, p_val_PoP = scipy.stats.pearsonr(results_df_PoP['True Age PoP'],results_df_PoP['Predicted Age PoP'])

"""BAG"""
results_df_PoP['BAG PoP'] = results_df_PoP['Predicted Age PoP'] - results_df_PoP['True Age PoP']

fig100=plt.figure('scatter chronological Age vs BAG PoP set',figsize=(16,16))
ax11=sns.jointplot(data=results_df_PoP, x="True Age PoP", y="BAG PoP", 
                kind="kde",fill=True, joint_kws={'alpha': 0.5} ,
                    color='#D45814')  # peru
x1=35
y1=-20
x2=100
y2=20

ax11.ax_joint.set_xlim(x1,x2)
ax11.ax_joint.set_ylim(y1,y2)

ax11.ax_joint.plot([x1, x2], [0, 0], color='#A21441', linestyle='-', linewidth=2)

ax11.ax_joint.tick_params(axis='both', labelsize=20)
sns.regplot(data=results_df_PoP, x='True Age PoP', y='BAG PoP',
    scatter=False, color='black', line_kws={'linestyle':'--'}, ax=ax11.ax_joint)

#calculate slope and intercept of regression equation
slope_BAG_PoP, intercept_BAG_PoP, r_BAG_PoP, p_BAG_PoP, sterr_BAG_PoP = scipy.stats.linregress( x=ax11.ax_joint.get_lines()[1].get_xdata(),
                                                                y=ax11.ax_joint.get_lines()[1].get_ydata())


# Add regression equation to the joint plot
ax11.ax_joint.text(70, 6, 'y = ' + str(round(intercept_BAG_PoP, 3)) + ' + ' + str(round(slope_BAG_PoP, 3)) + 'x')

plt.grid()
# plt.show()
plt.savefig(Path_to_Save_Results+'/extra_results' + '/7_PoP_set_BAG_before_Bias_correction.png')
plt.close(fig100)
plt.close('all')
del fig100
del ax11
"""Bias correction"""

results_df_PoP['Corrected predicted Age PoP'] = (results_df_PoP['Predicted Age PoP']-intercept_train)/slope_train

"""plot bais corrected brain age"""
fig100=plt.figure('scatter corrected predicted Age vs chronological Age PoP set',figsize=(16,16))
ax11=sns.jointplot(data=results_df_PoP, x="True Age PoP", y="Corrected predicted Age PoP", 
                kind="kde",fill=True, joint_kws={'alpha': 0.5} ,
                    color='#D45814')  # peru
x1=35
y1=35
x2=100
y2=100

ax11.ax_joint.set_xlim(x1,x2)
ax11.ax_joint.set_ylim(y1,y2)

# ax11.ax_joint.plot([x1,x2], [y1,y2], 'r-', linewidth = 2)
ax11.ax_joint.plot([x1, x2], [y1, y2], color='#A21441', linestyle='-', linewidth=2)

ax11.ax_joint.tick_params(axis='both', labelsize=20)
sns.regplot(data=results_df_PoP, x='True Age PoP', y='Corrected predicted Age PoP',
    scatter=False, color='black', line_kws={'linestyle':'--'}, ax=ax11.ax_joint)

#calculate slope and intercept of regression equation
slope_c_PoP, intercept_c_PoP, r_c_PoP, p_c_PoP, sterr_c_PoP = scipy.stats.linregress( x=ax11.ax_joint.get_lines()[1].get_xdata(),
                                                                y=ax11.ax_joint.get_lines()[1].get_ydata())


# Add regression equation to the joint plot
ax11.ax_joint.text(50, 80, 'y = ' + str(round(intercept_c_PoP, 3)) + ' + ' + str(round(slope_c_PoP, 3)) + 'x')
# plt.title('Brain Age prediction on healthy set (PoP set)')
plt.grid()
# plt.show()
plt.savefig(Path_to_Save_Results+'/extra_results' + '/8_PoP_set_after_Bias_correction.png')
plt.close(fig100)
plt.close('all')
del fig100
del ax11

MAE_c_PoP = mean_absolute_error(results_df_PoP['True Age PoP'],results_df_PoP['Corrected predicted Age PoP'])
MSE_c_PoP = mean_squared_error(results_df_PoP['True Age PoP'],results_df_PoP['Corrected predicted Age PoP'])
corr_c_PoP, p_val_c_PoP = scipy.stats.pearsonr(results_df_PoP['True Age PoP'],results_df_PoP['Corrected predicted Age PoP'])

"""bais corrected BAG"""
results_df_PoP['Corrected BAG PoP'] = results_df_PoP['Corrected predicted Age PoP'] - results_df_PoP['True Age PoP']
fig100=plt.figure('scatter chronological Age vs corrected BAG PoP set',figsize=(16,16))
ax11=sns.jointplot(data=results_df_PoP, x="True Age PoP", y="Corrected BAG PoP", 
                kind="kde",fill=True, joint_kws={'alpha': 0.5} ,
                    color='#D45814')  # peru
x1=35
y1=-20
x2=100
y2=20

ax11.ax_joint.set_xlim(x1,x2)
ax11.ax_joint.set_ylim(y1,y2)

ax11.ax_joint.plot([x1, x2], [0, 0], color='#A21441', linestyle='-', linewidth=2)

ax11.ax_joint.tick_params(axis='both', labelsize=20)
sns.regplot(data=results_df_PoP, x='True Age PoP', y='Corrected BAG PoP',
    scatter=False, color='black', line_kws={'linestyle':'--'}, ax=ax11.ax_joint)

#calculate slope and intercept of regression equation
slope_cBAG_PoP, intercept_cBAG_PoP, r_cBAG_PoP, p_cBAG_PoP, sterr_cBAG_PoP = scipy.stats.linregress( x=ax11.ax_joint.get_lines()[1].get_xdata(),
                                                                y=ax11.ax_joint.get_lines()[1].get_ydata())


# Add regression equation to the joint plot
ax11.ax_joint.text(70, 6, 'y = ' + str(round(intercept_cBAG_PoP, 3)) + ' + ' + str(round(slope_cBAG_PoP, 3)) + 'x')

plt.grid()
# plt.show()
plt.savefig(Path_to_Save_Results+'/extra_results' + '/9_Healthy_PoP_set_BAG_after_Bias_correction.png')
plt.close(fig100)
plt.close('all')
del fig100
del ax11

# %% extra Saves

scores.to_csv(Path_to_Save_Results+'/extra_results'+'/scores.csv')

results_df_train.to_csv(Path_to_Save_Results+'/extra_results'+'/train.csv')
results_df_test.to_csv(Path_to_Save_Results+'/extra_results'+'/test.csv')
results_df_PoP.to_csv(Path_to_Save_Results+'/extra_results'+'/PoP.csv')

slope_train, intercept_train



Summary=pd.DataFrame({'Names':['Train set', 
                               'Test set (Brain Age)', 
                               'Test set (corrected Brain Age)', 
                               'PoP set (Brain Age)', 'PoP set (corrected Brain Age)'],
                      'MAE':[MAE_train, MAE_test, MAE_c_test, MAE_PoP ,MAE_c_PoP],
                      'MSE':[MSE_train, MSE_test, MSE_c_test, MSE_PoP, MSE_c_PoP],
                      'Corr':[corr_train, corr_test, corr_c_test, corr_PoP, corr_c_PoP],
                      'Pval':[p_val_train, p_val_test, p_val_c_test, p_val_PoP, p_val_c_PoP],
                      'Slope':[slope_train, np.NAN, np.NAN, np.NAN, np.NAN],
                      'Intercept':[intercept_train, np.NAN, np.NAN, np.NAN, np.NAN]
})
Summary.to_csv(Path_to_Save_Results+'/extra_results'+'/Summary.csv')
# %%

sys.stdout.write(RED)
print(
      "\n\n================================================="
      "\nProgramm ended\n"
      "=================================================\n\n"
      )
sys.stdout.write(RESET)
# %%
