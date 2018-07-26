#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 10:38:29 2018

@author: jan
"""

#Predict House Price Project

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('kc_house_data.csv')

X = dataset.iloc[:, 3:15].values
X = np.append(X, dataset.iloc[:, 17:21].values, axis = 1) #axis 1 olması kolon birleştirmesi anlamına geliyor

y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)

'''# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(y_train.size, 1))
y_test = sc_y.transform(y_test.reshape(y_test.size, 1))'''

# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state = 1)
regressor.fit(X_train, y_train)

# Predicting a new result
y_pred = regressor.predict(X_test)

# Applying k-Fold Cross Validation 
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = regressor, X = X_train , y = y_train, cv = 10) 
accuracies.mean() 
accuracies.std() 

'''# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'n_estimators': [300], 'criterion': ['mse', 'mae'], 'max_features' : ['auto', 'sqrt', 'log2', None]}]
grid_search = GridSearchCV(estimator = regressor,
                           param_grid = parameters,
                           scoring = 'r2',
                           cv = 10,
                           n_jobs = -1) 
grid_search = grid_search.fit(X_train, y_train) 
best_accuracy = grid_search.best_score_  
best_parameters = grid_search.best_params_ '''

# Computing R-Squared
from sklearn.metrics import r2_score
rSquare = r2_score(y_test, y_pred)















