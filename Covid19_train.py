#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 09:26:39 2022

@author: angela
"""

#%% IMPORTS 

from sklearn.metrics import mean_absolute_percentage_error as MAPE
from sklearn.metrics import mean_absolute_error,mean_squared_error 
from tensorflow.keras.callbacks import TensorBoard,EarlyStopping
from Covid_module import ModelDevelopment, ModelEvaluation
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential, Input
from tensorflow.keras.utils import plot_model

import matplotlib.pyplot as plt 
import tensorflow as tf
import pandas as pd 
import numpy as np 
import datetime
import pickle
import os 

#%% CONSTANTS

CSV_PATH_TRAIN = os.path.join(os.getcwd(),'datasets','cases_malaysia_train.csv')

CSV_PATH_TEST = os.path.join(os.getcwd(),'datasets','cases_malaysia_test.csv')

MMS_PATH_X = os.path.join(os.getcwd(),'model','_mms.pkl')

LOGS_PATH = os.path.join(os.getcwd(),
                         'logs',
                         datetime.datetime.now().strftime('%y%m%d-%H%M%S'))


#%% 1) Data Loading 

df_train = pd.read_csv(CSV_PATH_TRAIN,na_values=(' ','?'))

df_test = pd.read_csv(CSV_PATH_TEST,na_values=(' ','?'))


#%% 2) Data Inspection 


# Train dataset 
df_train.info()  # total train data entries > 680
df_train.isna().sum() 
# found null value in cluster_import,cluster_religious,cluster_community,
#cluster_highRisk,cluster_education,cluster_detentionCentre,cluster_workplace

# Highlights: inspection of datsets from variable explorer found out the data contains empty space and ? entry. 
# roll back to data loading to state the na_values 
# after stationg na_values from data loading, now cases_new has 12 NaNs 
df_train.head()
df_train.tail()
pd.set_option('display.max_columns',None)
df_train.describe().T


# Test dataset 
df_test.info() # test dataset has total of 100 data entries 
df_test.isna().sum() # one NaN spotted in cases_new 
pd.set_option('display.max_columns',None)
df_test.describe().T

#%% 3) Data Cleaning 

# fill up nan in cases_new column for time series data using interpolation method 

df_train = df_train.interpolate(method='polynomial', order=3)
df_train.isna().sum()

df_test = df_test.interpolate(method='polynomial', order=3)
df_test.isna().sum()

# to round up the value after interpolation 
df_train['cases_new'] = df_train.cases_new.round(0).astype(int)
df_test['cases_new'] = df_test.cases_new.round(0).astype(int)


#%% 4) Features selection 

X = df_train['cases_new']

mms = MinMaxScaler()
X = mms.fit_transform(np.expand_dims(X, axis=-1))

with open(MMS_PATH_X,'wb') as file: 
    pickle.dump(mms,file)

#
win_size = 30
X_train = []
y_train = []

for i in range(win_size,len(X)): 
    X_train.append(X[i-win_size:i])
    y_train.append(X[i])

X_train = np.array(X_train)
y_train = np.array(y_train)

#%% Test datasets 

dataset_cat = pd.concat((df_train['cases_new'],df_test['cases_new']))
len(dataset_cat)-len(df_test)-win_size

length_days = len(dataset_cat)-len(df_test)-win_size
tot_input = dataset_cat[length_days:]

XT = mms.transform(np.expand_dims(tot_input, axis=-1)) # only fit once transform the rest 

# 
X_test = []
y_test = []

for i in range(win_size, len(XT)):
    X_test.append(XT[i-win_size:i])
    y_test.append(XT[i])

X_test = np.array(X_test)
y_test = np.array(y_test)

#%% Model Development 

# LSTM is preferred when comes to time series data 

input_shape = np.shape(X_train)[1:]
len(np.unique(y_train))
nb_class = 1

md = ModelDevelopment()
model = md.simple_dl_model(input_shape,nb_class)

plot_model(model,show_shapes=True, show_layer_names=True)

#%% model compilation 
tf.random.set_seed(7)
model.compile(optimizer='adam', loss='mse',
              metrics=['mae','mse']) # metrics can be one or two

#callbacks 

tensorboard_callback = TensorBoard(log_dir=LOGS_PATH, histogram_freq=1)
early_callback = EarlyStopping(monitor = 'val_loss', patience=5)


# ModelCheckpoint
BEST_MODEL_PATH = os.path.join(os.getcwd(), 'models', 'best_model.h5')

mdc = ModelCheckpoint(BEST_MODEL_PATH,
                      monitor='val_mean_absolute_percentage_error',
                      save_best_only=True,
                      mode='min',
                      verbose=1)

#%% Model Training

hist = model.fit(X_train,y_train,
                 epochs=300,
                 callbacks=[tensorboard_callback,mdc],
                 validation_data=(X_test,y_test))


#%% Model Evaluation
print(hist.history.keys())

y_pred=model.predict(X_test)


print(hist.history.keys())

me = ModelEvaluation()
me.MSE_plot(hist)


actual_covid_cases = mms.inverse_transform(y_test)
predicted_covid_cases = mms.inverse_transform(y_pred)
me.Act_Pred(actual_covid_cases,predicted_covid_cases)


#%%%

print('MAE:',mean_absolute_error(actual_covid_cases,predicted_covid_cases))
print('MSE:',mean_squared_error(actual_covid_cases,predicted_covid_cases))
print('MAPE:',MAPE(y_test,y_pred))



