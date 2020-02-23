#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 12:25:11 2020

@author: quechecho
"""
#preparation des données

import numpy as np
import pandas as pd
import matplotlib.pyplot as ply




#importer le Dataset

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:,13].values

#nous avons crée un nouveau dataset où la colonne CreditScore est 
#maintenant l'indice 0. Ce que nous voulons c'est que les valeurs non 
#numériques deviennet numériques, I mean, qu'elles soient réencoder
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder_X_1 = LabelEncoder()#encodeur pour la colonne Geography
X[:,1] = label_encoder_X_1.fit_transform(X[:,1])
    
label_encoder_X_2 = LabelEncoder()#encodeur pour la colonne Gender
X[:,2] = label_encoder_X_2.fit_transform(X[:,2])

#les valeurs de la colonne Gender sont Femme ou Homme en revanche
#les valeurs de la colonne Geography ne sont pas ordinale, I mean, on
#ne peut pas les ranger on ne peut pas dire que la France est plus 
#petit ou plus grand que l'Espagne ou l'Allemagne
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]
#découpez le dataset en test et training
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#mettre à l'echelle toutes les données pour faciliter le calcul
#données seront entre -1 et 1
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

import keras
from keras.models import Sequential#permet init un réseau de neurones
from keras.layers import Dense
from keras.layers import Dropout

#permet init un réseau de neurones appeller classifier parce que
#nous allons classifier

classifier = Sequential()

#ajout de la couche d'entrée et une couche cachée
#Dense(units= combien de neurones pour la couche cachée, activation=
#fonction d'activation, kernel_initializer= attribution des poids)
#vu que c'est la 1ere couche cachée il faut lui donnée la taille de la
#couche d'entrée input_dim= dans ce cas 11 neurones)
classifier.add(Dense(units=6, activation="relu", 
                     kernel_initializer="uniform", input_dim=11))
classifier.add(Dropout(rate=0.1))
#rate desactivation de neurones dans la couche(0.1 = 10%)
#ajouter une 2eme couche cachée

classifier.add(Dense(units=6, activation="relu", 
                     kernel_initializer="uniform"))
classifier.add(Dropout(rate=0.1))
#ajouter la couche de sortie dans units on remplace le 6 par le 1 car
#il n'y a qu'une seule variable à prédire, la fonction d'activation va changer
#car nous souhaitons une probabilité par rapport au fait si le client 
#va quitter la banque(fonction sigmoide il y a que deux catégories clients
#qui s'en vont ou qui reste si il y avait plus de catégories par exemple
#le pays Espagne France Allemagne Softmax) 
#et pas un oui ou non(fonction
#redresseur relu)

classifier.add(Dense(units=1, activation="sigmoid", 
                     kernel_initializer="uniform"))

#nous allons compiler 
classifier.compile(optimizer="adam", loss="binary_crossentropy", 
                   metrics=["accuracy"])


classifier.fit(X_train, y_train, batch_size = 10, epochs = 10)
# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cmeaz = confusion_matrix(y_test, y_pred)


#exercice
ar = sc.transform(np.array([[0,0,600,0,40,3,60000,2,1,1,50000]]))
#array avec deux crochets = array 2 dim avec une seule ligne



new_pred = classifier.predict(ar)
new_pred = (new_pred > 0.5)

#K-fold cross validation 
#divisée en 10 le data set et chaque division la diviser par 10 avec 
#9/10  pour train et 1/10 test

#une fois le traitement fait on va avoir 10 fois 1/10 de test. On calcule
#la moyenne et on obtient une meilleure précision. Si les valeurs sont
#very différentes haute variance else basse. Biais = difference de resultat
#prédit avec réellé

from keras.wrappers.scikit_learn import KerasClassifier
#import pour faire le lien entre keras et scikit_learn
from sklearn.model_selection import cross_val_score

def build_classifier(optimizer):
    classifier = Sequential()

#estoy cambiando de 6 a 12
    classifier.add(Dense(units=12, activation="relu", 
                     kernel_initializer="uniform", input_dim=11))



    classifier.add(Dense(units=12, activation="relu", 
                     kernel_initializer="uniform"))



    classifier.add(Dense(units=1, activation="sigmoid", 
                     kernel_initializer="uniform"))

    classifier.compile(optimizer=optimizer, loss="binary_crossentropy", 
                   metrics=["accuracy"])

    return classifier
#la sortie de cette def comprend les 10 valeurs de précisions.
#pour adapter le résultat de la def à cross_val_score on passe par KerasClassifier
    
classifier = KerasClassifier(build_fn=build_classifier, batch_size = 10, epochs = 100)

precisions = cross_val_score(estimator=classifier, X= X_train, y=y_train,cv=10,n_jobs=-1 )
#n_jobs = nombre de processeurs à utiliser quand -1 all les processeurs utilisés
moyenne = precisions.mean()
ecart_type = precisions.std()#variance

#drop out reduire le surapprentissage
# Tune
from sklearn.model_selection import GridSearchCV
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25, 36],
              'epochs': [100, 200],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_





