#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 16:50:58 2019

@author: ubuntu
"""
# amostragem do conjunto de dados
import pandas as pd
from sklearn.model_selection import train_test_split

dataset_log = pd.read_csv("Conversor de JSON/p2p-0.3-1-usuarios-nolle.csv")

dataset_log = dataset_log.iloc[:,:-1] # elimina ultima fila de "nan"
X = dataset_log.iloc[:,:-1] # copia todas as colunas menos o rótulo
Y = dataset_log['n'] # rótulos do dataset

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, stratify=Y, test_size=0.25) # 75% para treino e 25% para teste


dataset =
