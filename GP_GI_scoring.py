#!/usr/bin/env python
# coding: utf-8

# Importing required libraries

get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split 
from sklearn import metrics

# Loading single and double mutant fitness data

## import data file that has three columns: Wx, Wy, and Wxy

dataset = pd.read_csv('path-to-input-file/input.csv')
dataset.columns = ['Wx', 'Wy', 'Wxy']

# Training and test datasets generation

# Fold change

## function to calculate fold change between single and double mutant values
def FC_cal(DF):
    DF['FC_x_1'] = DF['Wx']/DF['Wxy']
    DF['FC_x_2'] = DF['Wxy']/DF['Wx']
    DF['FC_x'] = DF[['FC_x_1','FC_x_2']].min(axis=1)

    DF['FC_y_1'] = DF['Wy']/DF['Wxy']
    DF['FC_y_2'] = DF['Wxy']/DF['Wy']
    DF['FC_y'] = DF[['FC_y_1','FC_y_2']].min(axis=1)

    DF = DF.drop(['FC_x_1', 'FC_x_2', 'FC_y_1', 'FC_y_2'], axis=1)

    return DF

## adding fold change values to the data set
df = FC_cal(dataset)

## Training dataset generation

## number of instances for each value range. Several values can be tried.
num = 200
## difference between upper and lower values in a range. Several values can be tried.
range_value = 0.1

## first few rows of the data set
print('number of rows and columns in the data set: ', df.shape)
print()
print('A few instances of data set:')
print()
print(df.head())
print()

## iterating over different Wx mutant fitness value ranges and fold-change values to select training instances

df_x = []
for r in np.arange( (df['Wx'].min()), (df['Wx'].max()),   range_value):

    if len(df[(df['FC_x'] >= 0.9) & (df['Wx'].between(r,(r+0.1)))]) != 0:
        df_temp = ((df[(df['FC_x'] >= 0.9) 
                       & (df['Wx'].between(r,(r+0.1)))].sample(num, replace=True, random_state=40)))
        df_x.append(df_temp)
    else:
        continue

df_train_Wx = pd.concat(df_x)
df_train_Wx = df_train_Wx.drop_duplicates()

## iterating over different Wx mutant fitness value ranges and fold-change values to select training instances

df_y = []
for r in np.arange( (df['Wy'].min()), (df['Wy'].max()),   range_value):

    if len(df[(df['FC_y'] >= 0.9) & (df['Wy'].between(r,(r+0.1)))]) != 0:
        df_temp = ((df[(df['FC_y'] >= 0.9) 
                       & (df['Wy'].between(r,(r+0.1)))].sample(num, replace=True, random_state=41)))
        df_y.append(df_temp)   
    else:
        continue

df_train_Wy = pd.concat(df_y)
df_train_Wy = df_train_Wy.drop_duplicates()

## iterating over different Wxy mutant fitness value ranges and fold-change values to select training instances

df_x_y = []
for r in np.arange( ((df[['Wx','Wy']].min()).min()), ((df[['Wx','Wy']].max()).max()),   range_value):

    if len(df[(df['FC_x'] >= 0.8) & (df['FC_y'] >= 0.8) 
              & (df['Wx'].between(r,(r+0.1))) & (df['Wy'].between(r,(r+0.1)))]) != 0:
        df_temp = (df[(df['FC_x'] >= 0.8) & (df['FC_y'] >= 0.8) & (df['Wx'].between(r,(r+0.1))) 
                      & (df['Wy'].between(r,(r+0.1)))].sample(num, replace=True, random_state=41))
        df_x_y.append(df_temp)
        
    else:
        continue

df_train_Wx_Wy = pd.concat(df_x_y)
df_train_Wx_Wy = df_train_Wx_Wy.drop_duplicates()

## combining all selected instances to generate final training data set

df_training = pd.concat([df_train_Wx,df_train_Wy,
                         df_train_Wx_Wy]).drop_duplicates().reset_index(drop=True)

print('number of rows and columns in the training data set: ', df_training.shape)
print()
print('A few instances of training data set:')
print()
print(df_training.head())
print()

# Dummy data set

## here we show how a fake growth fitness data can be generated
## below, we extracted test data sets from this fake data
## growth fitness ranges of single and double mutants can be changed to see how results are impacted
## however, the real growth fitness data, as used for training data, should be used to make real test data
## please note that any instance that was used for training should not be part of test data set

df1 = pd.DataFrame()
df1['Wx'] = np.random.uniform(0.85,1.2, size=100000)
df1['Wy'] = np.random.uniform(0.5,1.5, size=100000)
df1['Wxy'] = np.random.uniform(0.4,1.8, size=100000)
df1['FC_x'] = np.random.uniform(0.4,1.0, size=100000)
df1['FC_y'] = np.random.uniform(0.4,1.0, size=100000)
print()
print('A few instances of training data:')
print()
print(df1.head())
print()

# Test negative examples

df_neg = pd.DataFrame()
df_temp_1 = (df1[(df1['FC_x'] >= 0.9) & (df1['FC_y'] >= 0.9)]).sample(200, replace=True, random_state=42)
df_temp_2 = (df1[( (df1['Wx'] - df1['Wy']) >= 0.4 ) & (df1['FC_y'] >= 0.8)]).sample(200,  replace=True, random_state=42)
df_temp_3 = (df1[( (df1['Wy'] - df1['Wx']) >= 0.4 ) & (df1['FC_x'] >= 0.8)]).sample(200,  replace=True, random_state=42)
df_test_neg = pd.concat([df_temp_1,df_temp_2,df_temp_3]).drop_duplicates().reset_index(drop=True)
# df_test_neg.head()

# Test positive examples
df_pos = pd.DataFrame()
df_temp_1 = (df1[(df1['FC_x'] <= 0.7) & (df1['FC_y'] <= 0.7)]).sample(200, replace=True, random_state=42)
df_temp_2 = (df1[( (df1['Wx'] - df1['Wy']) >= 0.4 ) & (df1['FC_x'] >= 0.8)]).sample(200,  replace=True, random_state=42)
df_temp_3 = (df1[( (df1['Wy'] - df1['Wx']) >= 0.4 ) & (df1['FC_y'] >= 0.8)]).sample(200,  replace=True, random_state=42)
df_test_pos = pd.concat([df_temp_1,df_temp_2,df_temp_3]).drop_duplicates().reset_index(drop=True)

## Final test dataset

df_test = pd.concat([df_test_neg, df_test_pos]).drop_duplicates().reset_index(drop=True)
print('number of rows and columns in the test data set: ', df_test.shape)
print()
print('A few instances of test data set:')
print()
print(df_test.head())
print()

# Loading sklearn libraries for Gaussian Processes

from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RBF
from sklearn.gaussian_process import GaussianProcessRegressor

# Kernel for GP

## here we show one way to use kernel function
## other kernel options can also be tried from sklearn

kernel = ConstantKernel() + Matern(length_scale=2, nu=3/2) + WhiteKernel(noise_level=1)

# Data preprocessing

## Wx, Wy and Wxy are donor, recipient and double mutant normalized colony sizes, respectively.

X = df_training[['Wx', 'Wy']]
y = df_training['Wxy'].values.reshape(-1,1)

## data split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Training and testing of model

## training the algorithm

gp = gaussian_process.GaussianProcessRegressor(kernel=kernel)
gp.fit(X_train, y_train) 

## Making predictions on test data

y_pred = gp.predict(X_test)

## Error metrics calculation

obs_and_pred = pd.DataFrame({'Wxy_obs': y_test.flatten(), 'Wxy_pred': y_pred.flatten()})

mae = metrics.mean_absolute_error(obs_and_pred['Wxy_obs'], obs_and_pred['Wxy_pred'])
rmse = np.sqrt(metrics.mean_squared_error(obs_and_pred['Wxy_obs'], obs_and_pred['Wxy_pred']))

print("Mean Absolute Error = ", mae)
print("Root Mean Square Error = ", rmse)
