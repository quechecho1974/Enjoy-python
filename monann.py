#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 16:50:28 2020

@author: quechecho
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as ply

ds2014 = pd.read_csv('2014_Financial_Data.csv')
ds2014_nan = ds2014.fillna(0)
del ds2014_nan['Unnamed: 0']
del ds2014_nan['Sector']
float_col = ds2014_nan.select_dtypes(include=['float64']) # This will select float columns only
# list(float_col.columns.values)
for col in float_col.columns.values:
                ds2014_nan[col] = ds2014_nan[col].astype('int64')


ds2014_nan
len(ds2014_nan)
ds2014_nan.head(2)
ds2015 = pd.read_csv('2015_Financial_Data.csv')
ds2015_nan = ds2015.fillna(0)
del ds2015_nan['Unnamed: 0']
del ds2015_nan['Sector']
float_col = ds2015_nan.select_dtypes(include=['float64']) # This will select float columns only
# list(float_col.columns.values)
for col in float_col.columns.values:
                ds2015_nan[col] = ds2015_nan[col].astype('int64')
ds2015_nan
ds2016 = pd.read_csv('2016_Financial_Data.csv')
ds2016_nan = ds2016.fillna(0)
del ds2016_nan['Unnamed: 0']
del ds2016_nan['Sector']
ds2016_nan
ds2017 = pd.read_csv('2017_Financial_Data.csv')
ds2017_nan = ds2017.fillna(0)
del ds2017_nan['Unnamed: 0']
del ds2017_nan['Sector']
ds2017_nan
ds2018 = pd.read_csv('2018_Financial_Data.csv')
ds2018_nan = ds2018.fillna(0)
del ds2018_nan['Unnamed: 0']
del ds2018_nan['Sector']
ds2018_nan

ds1415 = ds2014_nan.append(ds2015_nan, ignore_index = True) 
ds1415.fillna(0)
del ds1415['2016 PRICE VAR [%]']
ds1415

Big_DS_14_15 = ds2014.append(ds2015, ignore_index = True)
Big_DS_14_16 = Big_DS_14_15.append(ds2016, ignore_index = True)

Big_DS_14_17 = Big_DS_14_16.append(ds2017, ignore_index = True)

Big_DS_14_18 = Big_DS_14_17.append(ds2018, ignore_index = True)

Big_full = Big_DS_14_18.fillna(0)

Xa = ds1415.iloc[:, 0:223].values
ya = ds1415.iloc[:,222].values

from sklearn.model_selection import train_test_split
Xa_train, Xa_test, ya_train, ya_test = train_test_split(Xa, ya, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
Xa_train = sc.fit_transform(Xa_train)
Xa_test = sc.transform(Xa_test)

import keras
from keras.models import Sequential#permet init un réseau de neurones
from keras.layers import Dense
from keras.layers import Dropout

classifiera = Sequential()

classifiera.add(Dense(units=12, activation="relu", 
                     kernel_initializer="uniform", input_dim=223))
classifiera.add(Dropout(rate=0.1))

classifiera.add(Dense(units=12, activation="relu", 
                     kernel_initializer="uniform"))
classifiera.add(Dropout(rate=0.1))

classifiera.add(Dense(units=1, activation="sigmoid", 
                     kernel_initializer="uniform"))

#nous allons compiler 
classifiera.compile(optimizer="adam", loss="binary_crossentropy", 
                   metrics=["accuracy"])

classifiera.fit(Xa_train, ya_train, batch_size = 10, epochs = 100)

y_predaa = classifiera.predict(Xa_test)
y_predaa = (y_predaa > 0.5)

from sklearn.metrics import confusion_matrix
cmeazaa = confusion_matrix(ya_test, y_predaa)























