#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: Mostafa Mahdipour
"""

# %%

import sys

import numpy as np 
import pandas as pd # to use dataframe
import matplotlib.pyplot as plt # to make plots
# import scipy # to calculate correlation
import seaborn as sns # to make plots
import scipy # to calculate correlation
from sklearn.metrics import mean_absolute_error # to calculate MAE

# !pip install -U julearn
from julearn import run_cross_validation
from julearn.utils import configure_logging
from sklearn.kernel_ridge import KernelRidge
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
sys.stdout.write(CYAN)
print('Jlearn info\n')
sys.stdout.write(RESET)
# To log information
configure_logging(level='INFO')
#%% loading needed *csv files
sys.stdout.write(CYAN)
print("\n Loading CAT features of Healthy group and Population")
sys.stdout.write(RESET)

Cat_Features=pd.read_csv('../1_DATA/CATs_Age_Healthy/Age_Schaefer_17Network_200_UKB_Tian_Healthy.csv') #healthy
Cat_Features.drop(columns=['Unnamed: 0'],inplace=True)
Cat_Features0=pd.read_csv('../1_DATA/CATs_Age/Age_Schaefer_17Network_200_UKB_Tian.csv') #Population
Cat_Features0.drop(columns=['Unnamed: 0'],inplace=True)
Cat_Features_PoP=Cat_Features0[~Cat_Features0.subject_ID.isin(Cat_Features.subject_ID)] # Population-healthy

sys.stdout.write(CYAN)
print("\n Loading is finished")
sys.stdout.write(RESET)

# %% Outlier treatment

'''

Outlier treatment
    Outliers are extreme values in the data which are far away from most of the
    values. You can see them as the tails in the histogram.

    Outlier must be treated one column at a time. As the treatment will be slightly
    different for each column.

    Why I should treat the outliers?

    Outliers bias the training of machine learning models. As the algorithm tries
    to fit the extreme value, it goes away from majority of the data.

    There are below two options to treat outliers in the data.

        Option-1: Delete the outlier Records. Only if there are just few rows lost.
        Option-2: Impute the outlier values with a logical business value


'''

plt.figure("GM_Before_After",figsize=(18,10))

plt.subplot(2, 1, 1)
plt.hist(Cat_Features['GM'],color=('r'),bins=30)
title_obj=plt.title("GM Histogram plot Before and After Outlier treatment")
plt.setp(title_obj, color='r')
plt.grid()
plt.subplot(2,1,2)
# plt.figure("GM_Before_After").title('Hist plot GM Before and After Outlier treatment')
sys.stdout.write(CYAN)
print('\n Outlier treatment based on GM histogram\n')
print('\n excluding participants with GM lower than 400 and biger than 850\n')

out_low=Cat_Features.loc[Cat_Features['GM']<=400]
out_high=Cat_Features.loc[Cat_Features['GM']>=850]

# out_high=Cat_Features[Cat_Features['GM'].gt(850)]

frames = [out_low, out_high]

GM_Outliers_train = pd.concat(frames)

sys.stdout.write(GREEN)

print("{0:<7} {1:<20}".format('Index', 'GM_Outliers_Value'))
sys.stdout.write(RESET)
print(GM_Outliers_train['GM'])
print(GM_Outliers_train['GM'].shape)
Cat_Features_GMcorrected=Cat_Features.drop(GM_Outliers_train.index)
plt.hist(Cat_Features_GMcorrected['GM'], bins=30,color=('g'))
plt.grid()
plt.show()




# Population
plt.figure("GM_Before_After1",figsize=(18,10))

plt.subplot(2, 1, 1)
plt.hist(Cat_Features_PoP['GM'],color=('r'),bins=30)
title_obj=plt.title("GM Histogram plot Before and After Outlier treatment")
plt.setp(title_obj, color='r')
plt.grid()
plt.subplot(2,1,2)
# plt.figure("GM_Before_After").title('Hist plot GM Before and After Outlier treatment')
sys.stdout.write(CYAN)
print('\n Outlier treatment based on GM histogram\n')
print('\n excluding participants with GM lower than 400 and biger than 850\n')

out_low=Cat_Features_PoP.loc[Cat_Features_PoP['GM']<=400]
out_high=Cat_Features_PoP.loc[Cat_Features_PoP['GM']>=850]


frames = [out_low, out_high]

GM_Outliers = pd.concat(frames)

sys.stdout.write(GREEN)

print("{0:<7} {1:<20}".format('Index', 'GM_Outliers_Value'))
sys.stdout.write(RESET)
print(GM_Outliers['GM'])
print(GM_Outliers['GM'].shape)
Cat_Features_PoP_GMcorrected=Cat_Features_PoP.drop(GM_Outliers.index)
plt.hist(Cat_Features_PoP_GMcorrected['GM'], bins=30,color=('g'))
plt.grid()
plt.show()
# Cat_Features_PoP_GMcorrected.drop(columns=['session.1'],inplace=True)
# Cat_Features_GMcorrected.drop(columns=['session.1'],inplace=True)
#%% JuLearn ridge

from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
# Check the columns and rows name
data_df=Cat_Features_GMcorrected
# data_df=Cat_Features
sys.stdout.write(CYAN)
print('Column names: ', data_df.columns ,'\n')
print('Index names: ', data_df.index, '\n')
sys.stdout.write(RESET)
X_list = list(set(data_df.columns) - set(['subject_ID', 'session', 'NCR','ICR', 
                                          'IQR', 'TIV', 'GM','WM', 'CSF','WMH',
                                          'TSA','Age','Gender'])) # features
y = 'Age' # target

data_df.sort_values(by='Age', inplace=True, ignore_index=True)

# Create train and test split
qc_data = pd.cut(data_df['Age'].tolist(), bins=5, precision=1)  # Age bins to create straified train-test split
print('Age_bins', qc_data.categories, 'Age_codes', qc_data.codes)
train_df, test_df = train_test_split(data_df, stratify=qc_data.codes, test_size=0.25)  # create train and test data

rand_seed = 94
# var_threshold = 1e-5  # threshold used for variance thresholding on features

# Define pre-processing steps
# preprocess_X = ['select_variance', 'zscore']

# Define model and parameters to be used for 'run_cross_validation' function
# For Kernel Ridge, import and define kernel ridge model from sklearn as it is not present in julearn
# model_name = KernelRidge()

# Define parameter space for hyperparameter tuning (here 5fold CV will be used as 'cv':5)
model_params = {'ridge__alpha':[0.0001,0.001, 0.05, 0.01, 0, 0.1, 0.5, 1, 2, 3, 4,
                              5, 10,50,100],'cv': 10, 'search':'grid'} # for gridsearch

# create Age bins on train data to do stratified Kfold CV
qc_train = pd.cut(train_df['Age'].tolist(), bins=5, precision=1) # Age bins to create straified CV folds
print('Age_bins', qc_train.categories, 'Age_codes', qc_train.codes)

# Define type of CV to estimate generalization performance
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=rand_seed).split(train_df, qc_train.codes)

# run 'run_cross_validation' function
# Note that here we are performing outer 10 times 5 fold CV to estimate generalization performance,
# with internal 10 fold CV for hyperparmeter tuning and a final model trained on whole train data
scores, model = run_cross_validation(X=X_list, y=y, data=train_df,
                                     problem_type='regression', model='ridge', cv=cv,
                             return_estimator='final', model_params=model_params, seed=rand_seed,
                                     scoring=
                             ['neg_mean_absolute_error', 'neg_mean_squared_error','r2'])

models = model.best_estimator_
print('best model', model.best_estimator_)
print('best para', model.best_params_)
#%%
# Get prediction for test data
y_true = test_df[y]
y_pred = model.predict(test_df[X_list]).ravel()
mae = mean_absolute_error(y_true, y_pred)
corr, p = scipy.stats.pearsonr(y_pred, y_true)
sys.stdout.write(CYAN)
print('Test MAE: ', mae, 'Test Corr: ', corr)
sys.stdout.write(RESET)

# Prepare results to plot

results_df_test = pd.DataFrame({'True age': y_true, 'Predicted age': y_pred})
print(results_df_test.head(10))

# Generate scatter plot
fig100=plt.figure('scatter predicted brain age vs chronological age',figsize=(16,8))
ax2=fig100.add_subplot(1,2,1)
sns.scatterplot(data=results_df_test, x='True age', y='Predicted age', s=50,color = 'peru',edgecolor = 'black',alpha = .7)
sns.lineplot(data=results_df_test, x='True age', y='True age', color= 'r')
# plt.title('MAE: ' + str(mae) + '   CORR: ' + str(corr))
from sklearn import linear_model
reg_test=linear_model.LinearRegression()

input1_realAage_test=[results_df_test['True age']]
input1_realAage_test=np.transpose(input1_realAage_test)

output1_predictAage_test=[results_df_test['Predicted age']]
output1_predictAage_test=np.transpose(output1_predictAage_test)
reg_test.fit(input1_realAage_test,output1_predictAage_test)
m_test=reg_test.coef_[0]
b_test=reg_test.intercept_
# print("slope=",m, "intercept=",b)
# plt.scatter(input1_realAage,output1_predictAage,color='blue')
predicted_values = [reg_test.coef_[0] * ii + reg_test.intercept_ for ii in input1_realAage_test]
plt.plot(input1_realAage_test, predicted_values, 'k', linewidth=3, linestyle=':')
plt.xlim([40, 100])
plt.ylim([40, 100])
plt.grid()
a = '{0:.3g}'.format(mae)
b = '{0:.3g}'.format(corr)
c = '{0:.3g}'.format(m_test[0])
d = '{0:.3g}'.format(b_test[0])
ax2.title.set_text('MAE: ' + str(a) + '   CORR: ' + str(b)+'   SLOPE='+ str(c)+'   INTERCEPT='+str(d))

# bias correction
ax3=fig100.add_subplot(1,2,2)
y_pred_train = model.predict(train_df[X_list])
y_true_train = train_df[y]
results_df_train = pd.DataFrame({'True age t': y_true_train, 'Predicted age t': y_pred_train})


reg_train=linear_model.LinearRegression()
input1_realAage_train=[results_df_train['True age t']]
input1_realAage_train=np.transpose(input1_realAage_train)

output1_predictAage_train=[results_df_train['Predicted age t']]
output1_predictAage_train=np.transpose(output1_predictAage_train)

reg_train.fit(input1_realAage_train,output1_predictAage_train)
m_train=reg_train.coef_[0]
b_train=reg_train.intercept_
print("slope=",m_train, "intercept=",b_train)

corrected_predicted_age_test=(y_pred-b_train)/m_train
# plt.scatter(input1_realAage,output1_predictAage,color='blue')
# predicted_values_line = [reg.coef_[0] * ii + reg.intercept_ for ii in y_true]
# fig2,plt.plot(y_true, predicted_values_line, 'k')
# plt.xlabel("treu age")
# plt.ylabel("predicted age")


results_df_test_corrected=results_df_test
results_df_test_corrected['corrected Predicted age']=corrected_predicted_age_test
mae_c1 = mean_absolute_error(y_true, corrected_predicted_age_test)
corr_c1, p_c1 = scipy.stats.pearsonr(corrected_predicted_age_test, y_true)

# fig3=plt.figure('scatter predicted brain age vs chronological age')
sns.scatterplot(data=results_df_test_corrected, x='True age', y='corrected Predicted age', s=50,edgecolor = 'black',alpha = .5)
sns.lineplot(data=results_df_test_corrected, x='True age', y='True age', color= 'r',linewidth=3)
# fig3,plt.title('MAE: ' + str(mae_c1) + '   CORR: ' + str(corr_c1))
reg_test_c=linear_model.LinearRegression()

input1_realAage_test_c=[results_df_test_corrected['True age']]
input1_realAage_test_c=np.transpose(input1_realAage_test_c)

output1_predictAage_test_c=[results_df_test_corrected['corrected Predicted age']]
output1_predictAage_test_c=np.transpose(output1_predictAage_test_c)
reg_test_c.fit(input1_realAage_test_c,output1_predictAage_test_c)
m_test_c=reg_test_c.coef_[0]
b_test_c=reg_test_c.intercept_
# print("slope=",m, "intercept=",b)
# plt.scatter(input1_realAage,output1_predictAage,color='blue')
predicted_values = [reg_test_c.coef_[0] * ii + reg_test_c.intercept_ for ii in input1_realAage_test_c]
plt.plot(input1_realAage_test_c, predicted_values, 'k', linewidth=3, linestyle=':')
plt.xlim([40, 100])
plt.ylim([40, 100])
plt.grid()
a = '{0:.3g}'.format(mae_c1)
b = '{0:.3g}'.format(corr_c1)
c = '{0:.3g}'.format(m_test_c[0])
d = '{0:.3g}'.format(b_test_c[0])
ax3.title.set_text('MAE: ' + str(a) + '   CORR: ' + str(b)+'   SLOPE='+ str(c)+'   INTERCEPT='+str(d))
a = '{0:.3g}'.format(m_train[0])
b = '{0:.3g}'.format(b_train[0])
fig100.suptitle('replication 0  main SLOPE='+ str(a)+'  main INTERCEPT='+str(b))
plt.show()
#%% slope and 
assa=plt.figure('Train1')
sns.scatterplot(data=results_df_train, x='True age t', y='Predicted age t', s=50,color = 'c',edgecolor = 'black',alpha = .5)
sns.lineplot(data=results_df_train, x='True age t' , y='True age t', color= 'r',linewidth=3)
input1_realAage_train=[results_df_train['True age t']]
input1_realAage_train=np.transpose(input1_realAage_train)
predicted_values = [reg_train.coef_[0] * ii + reg_train.intercept_ for ii in input1_realAage_train]
plt.plot(input1_realAage_train, predicted_values, 'k', linewidth=3, linestyle=':')
plt.grid()
plt.show()
#%% gap calculation on test set
BrainAgeGap_test=results_df_test
gap_=results_df_test['Predicted age']-results_df_test['True age']
gap1=np.array(gap_)
BrainAgeGap_test['GAP']=gap1

g = sns.JointGrid(data=BrainAgeGap_test, x="True age", y="GAP")
g.plot_joint(sns.scatterplot)

g.plot_marginals(sns.histplot,kde=True,color = 'red')
g.plot(sns.regplot,sns.histplot,color = 'darkorange')

regline = g.ax_joint.get_lines()[0]
regline.set_color('red')
regline.set_zorder(5)
plt.plot([45, 85], [0, 0], linewidth=2,color = 'k',linestyle=':')
plt.grid()
plt.show()


mae_PoPgt=mean_absolute_error(BrainAgeGap_test['GAP'], np.zeros(len(BrainAgeGap_test['GAP'])))
corr_PoPgt,pPoPgt = scipy.stats.pearsonr(BrainAgeGap_test['GAP'], BrainAgeGap_test['corrected Predicted age'])
a = '{0:.3g}'.format(mae_PoPgt)
b = '{0:.3g}'.format(corr_PoPgt)



BrainAgeGap_test_c=results_df_test_corrected.filter(['True age','corrected Predicted age'])
gap_=results_df_test_corrected['corrected Predicted age']-results_df_test_corrected['True age']
gap1=np.array(gap_)
BrainAgeGap_test_c['corrected GAP']=gap1


g = sns.JointGrid(data=BrainAgeGap_test_c, x="True age", y="corrected GAP")
g.plot_joint(sns.scatterplot)

g.plot_marginals(sns.histplot,kde=True,color = 'red')
g.plot(sns.regplot,sns.histplot,color = 'blue')

regline = g.ax_joint.get_lines()[0]
regline.set_color('red')
regline.set_zorder(5)
plt.plot([45, 85], [0, 0], linewidth=2,color = 'k',linestyle=':')
plt.grid()
plt.show()

mae_PoPgt=mean_absolute_error(BrainAgeGap_test_c['corrected GAP'], np.zeros(len(BrainAgeGap_test_c['corrected GAP'])))
corr_PoPgt,pPoPgt = scipy.stats.pearsonr(BrainAgeGap_test_c['corrected GAP'], BrainAgeGap_test_c['corrected Predicted age'])
a = '{0:.3g}'.format(mae_PoPgt)
b = '{0:.3g}'.format(corr_PoPgt)




#%% Population
y_pred_PoP = model.predict(Cat_Features_PoP_GMcorrected[X_list])
# y_pred_PoP = model.predict(Cat_Features_PoP[X])
y_true_PoP = Cat_Features_PoP_GMcorrected[y]
# y_true_PoP = Cat_Features_PoP[y]
mae_PoP = mean_absolute_error(y_true_PoP, y_pred_PoP)
corr_PoP, pPoP = scipy.stats.pearsonr(y_pred_PoP, y_true_PoP)

results_df_test_PoP = pd.DataFrame({'True age': y_true_PoP, 'Predicted age':y_pred_PoP })
print(results_df_test_PoP.head(10))

# Generate scatter plot
fig1000=plt.figure('scatter predicted brain age vs chronological age PoP',figsize=(16,8))
ax2=fig1000.add_subplot(1,2,1)
sns.scatterplot(data=results_df_test_PoP, x='True age', y='Predicted age', s=50,color = 'peru',edgecolor = 'black',alpha = .5)
sns.lineplot(data=results_df_test_PoP, x='True age', y='True age', color= 'r')
# plt.title('MAE: ' + str(mae) + '   CORR: ' + str(corr))
from sklearn import linear_model
reg_PoP=linear_model.LinearRegression()

input1_realAage_PoP=[results_df_test_PoP['True age']]
input1_realAage_PoP=np.transpose(input1_realAage_PoP)

output1_predictAage_PoP=[results_df_test_PoP['Predicted age']]
output1_predictAage_PoP=np.transpose(output1_predictAage_PoP)
reg_PoP.fit(input1_realAage_PoP,output1_predictAage_PoP)
m_PoP=reg_PoP.coef_[0]
b_PoP=reg_PoP.intercept_
# print("slope=",m, "intercept=",b)
# plt.scatter(input1_realAage,output1_predictAage,color='blue')
predicted_values = [reg_PoP.coef_[0] * ii + reg_PoP.intercept_ for ii in input1_realAage_PoP]
plt.plot(input1_realAage_PoP, predicted_values, 'k', linewidth=3, linestyle=':')
plt.xlim([40, 90])
plt.ylim([40, 90])
plt.grid()
a = '{0:.3g}'.format(mae_PoP)
b = '{0:.3g}'.format(corr_PoP)
c = '{0:.3g}'.format(m_PoP[0])
d = '{0:.3g}'.format(b_PoP[0])
ax2.title.set_text('MAE: ' + str(a) + '   CORR: ' + str(b)+'   SLOPE='+ str(c)+'   INTERCEPT='+str(d))

# bias correction
ax3=fig1000.add_subplot(1,2,2)
# y_pred_train = model.predict(train_df[X])
# y_true_train = train_df[y]
# results_df_train = pd.DataFrame({'True age t': y_true_train, 'Predicted age t': y_pred_train})


# reg_train=linear_model.LinearRegression()
# input1_realAage_train=[results_df_train['True age t']]
# input1_realAage_train=np.transpose(input1_realAage_train)

# output1_predictAage_train=[results_df_train['Predicted age t']]
# output1_predictAage_train=np.transpose(output1_predictAage_train)

# reg_train.fit(input1_realAage_train,output1_predictAage_train)
# m_train=reg_train.coef_[0]
# b_train=reg_train.intercept_
# print("slope=",m_train, "intercept=",b_train)
all_m=np.ones(9).reshape((3, 3))
corrected_predicted_age_PoP=np.true_divide(results_df_test_PoP['Predicted age']-b_train[0], (m_train[0]))

# (np.subtract(y_pred_PoP, b_train[0]))/(m_train[0])
# corrected_predicted_age_PoP=(y_pred_PoP - b_PoP)/m_PoP

# plt.scatter(input1_realAage,output1_predictAage,color='blue')
# predicted_values_line = [reg.coef_[0] * ii + reg.intercept_ for ii in y_true]
# fig2,plt.plot(y_true, predicted_values_line, 'k')
# plt.xlabel("treu age")
# plt.ylabel("predicted age")


results_df_PoP_corrected=results_df_test_PoP
results_df_PoP_corrected['corrected Predicted age']=corrected_predicted_age_PoP
mae_PoP = mean_absolute_error(y_true_PoP, corrected_predicted_age_PoP)
corr_PoP, p_PoP = scipy.stats.pearsonr(corrected_predicted_age_PoP, y_true_PoP)

# fig3=plt.figure('scatter predicted brain age vs chronological age')
sns.scatterplot(data=results_df_PoP_corrected, x='True age', y='corrected Predicted age',s=50,edgecolor = 'black',alpha = .3)
sns.lineplot(data=results_df_PoP_corrected, x='True age', y='True age', color= 'r',linewidth=3)
# fig3,plt.title('MAE: ' + str(mae_c1) + '   CORR: ' + str(corr_c1))
reg_PoP_c=linear_model.LinearRegression()

input1_realAage_PoP_c=[results_df_PoP_corrected['True age']]
input1_realAage_PoP_c=np.transpose(input1_realAage_PoP_c)

output1_predictAage_PoP_c=[results_df_PoP_corrected['corrected Predicted age']]
output1_predictAage_PoP_c=np.transpose(output1_predictAage_PoP_c)
reg_PoP_c.fit(input1_realAage_PoP_c,output1_predictAage_PoP_c)
m_PoP_c=reg_PoP_c.coef_[0]
b_PoP_c=reg_PoP_c.intercept_
# print("slope=",m, "intercept=",b)
# plt.scatter(input1_realAage,output1_predictAage,color='blue')
predicted_values = [reg_PoP_c.coef_[0] * ii + reg_PoP_c.intercept_ for ii in input1_realAage_PoP_c]
plt.plot(input1_realAage_PoP_c, predicted_values, 'k', linewidth=3, linestyle=':')
plt.xlim([20, 100])
plt.ylim([20, 130])
plt.grid()
a = '{0:.3g}'.format(mae_PoP)
b = '{0:.3g}'.format(corr_PoP)
c = '{0:.3g}'.format(m_PoP_c[0])
d = '{0:.3g}'.format(b_PoP_c[0])
ax3.title.set_text('MAE: ' + str(a) + '   CORR: ' + str(b)+'   SLOPE='+ str(c)+'   INTERCEPT='+str(d))
a = '{0:.3g}'.format(m_train[0])
b = '{0:.3g}'.format(b_train[0])
fig1000.suptitle('PoP main SLOPE='+ str(a)+'  main INTERCEPT='+str(b))
plt.show()

#%% gap calculation on PoP set
BrainAgeGap_PoP=results_df_test_PoP
gap_=results_df_test_PoP['Predicted age']-results_df_test_PoP['True age']
gap1=np.array(gap_)
BrainAgeGap_PoP['GAP']=gap1


g = sns.JointGrid(data=BrainAgeGap_PoP, x="True age", y="GAP")
g.plot_joint(sns.scatterplot)

g.plot_marginals(sns.histplot,kde=True,color = 'red')
g.plot(sns.regplot,sns.histplot,color = 'darkorange')

regline = g.ax_joint.get_lines()[0]
regline.set_color('red')
regline.set_zorder(5)
plt.plot([44, 85], [0, 0], linewidth=2,color = 'k',linestyle=':')
plt.grid()
plt.show()



mae_PoPg=mean_absolute_error(BrainAgeGap_PoP['GAP'], np.zeros(len(BrainAgeGap_PoP['GAP'])))
corr_PoPg,pPoPg = scipy.stats.pearsonr(BrainAgeGap_PoP['GAP'], BrainAgeGap_PoP['corrected Predicted age'])
a = '{0:.3g}'.format(mae_PoPg)
b = '{0:.3g}'.format(corr_PoPg)



BrainAgeGap_PoP_c=results_df_PoP_corrected.filter(['True age','corrected Predicted age'])
gap_=results_df_PoP_corrected['corrected Predicted age']-results_df_PoP_corrected['True age']
gap1=np.array(gap_)
BrainAgeGap_PoP_c['corrected GAP']=gap1
g = sns.JointGrid(data=BrainAgeGap_PoP_c, x="True age", y="corrected GAP")
g.plot_joint(sns.scatterplot)

g.plot_marginals(sns.histplot,kde=True,color = 'red')
g.plot(sns.regplot,sns.histplot,color = 'blue')

regline = g.ax_joint.get_lines()[0]
regline.set_color('red')
regline.set_zorder(5)
plt.plot([44, 85], [0, 0], linewidth=2,color = 'k',linestyle=':')

mae_PoPgc=mean_absolute_error(BrainAgeGap_PoP_c['corrected GAP'], np.zeros(len(BrainAgeGap_PoP_c['corrected GAP'])))
corr_PoPgc,pPoPgc = scipy.stats.pearsonr(BrainAgeGap_PoP_c['corrected GAP'], BrainAgeGap_PoP_c['corrected Predicted age'])
a = '{0:.3g}'.format(mae_PoPgc)
b = '{0:.3g}'.format(corr_PoPgc)


plt.grid()
plt.show()
# %% Ending statement
sys.stdout.write(RED)

print(
      "\n\n================================================="
      "\nprogram ended\nenjoy ;)\n"
      "=================================================\n\n"
      )
sys.stdout.write(RESET)