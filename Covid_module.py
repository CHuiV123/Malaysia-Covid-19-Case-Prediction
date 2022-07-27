#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 17:55:52 2022

@author: angela
"""

#%%

from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras import Sequential, Input
import matplotlib.pyplot as plt
import numpy as np

#%%
class ModelDevelopment:
    def simple_dl_model(self, input_shape,nb_class,nb_node=64,dropout_rate=0.2,
                        activation='relu'):
        """
        

        Parameters
        ----------
        input_shape : TYPE
            DESCRIPTION: input shape is the data shape of X train
        nb_class : TYPE
            DESCRIPTION: nb_class is the total of class we have in this dataset
        nb_node : TYPE, optional
            DESCRIPTION. The default is 64.
        dropout_rate : TYPE, optional
            DESCRIPTION. The default is 0.2.
        activation : TYPE, optional
            DESCRIPTION. The default is 'relu'.

        Returns
        -------
        None.

        """
        model = Sequential()
        model.add(Input(shape=(input_shape))) #LSTM,RNN,GRU only accepts 3D arrays
        model.add(LSTM(nb_node,return_sequences=(True)))#return_sequence to keep in 3D array 
        model.add(Dropout(dropout_rate))
        model.add(LSTM(nb_node))
        model.add(Dropout(dropout_rate))
        model.add(Dense(nb_class,activation))
        model.summary()

        return model

    
#%%

class ModelEvaluation():
    def MSE_plot(self,hist):
        plt.figure()
        plt.plot(hist.history['mse'])
        plt.plot(hist.history['val_mse'])
        plt.xlabel('epoch')
        plt.legend(['Training mse', 'Validation mse'])
        plt.grid()
        plt.show()
    
    def Act_Pred(self,actual_case,predicted_case): 
        plt.figure()
        plt.plot(actual_case,color='red')
        plt.plot(predicted_case,color='blue')
        plt.xlabel('Days')
        plt.ylabel('Covid 19 Cases')
        plt.legend(['Actual number of cases','Predicted number of cases'])
        plt.grid()
        plt.show()
