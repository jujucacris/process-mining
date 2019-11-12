#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#*******************************************
# VALIDACAO
# CROSS VALIDATION ESTRATIFICADO
# v1.0
#*******************************************

# Importacao de librarias
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold


# Definicao da arquitetura da rede
funcao_f='tan' # funcao de ativacao da camada de entrada
funcao_g='sig' # funcao de ativacao
nitmax=100 # numero de iterações maximo
alfa=0.8 # taxa de aprendizado
no=1148 # numero de nos da camada oculta
oMLP = cMLP(funcao_f,funcao_g,no) #definir a rede

# ler conjunto de dados
dataset = pd.read_csv("Conversor de JSON/p2p-0.3-1-nolle.csv")
dataset_X = np.array(dataset.iloc[:,:-2])
dataset_Y = np.array(dataset['n'])
#dataset_X = np.array([[1,0],[1,1],[0,0],[0,1],[1,0],[1,1],[0,0],[0,1]])
#dataset_Y = np.array([[0],[1],[0],[0],[0],[1],[0],[0]])

# Dividir o conjunto de treinamento em kFold cada um com partes para Train e Valid
k = 5

# folds contém conjuntos estratificados para trainamento e teste
folds = list(StratifiedKFold(n_splits=k, shuffle=True).split(dataset_X, dataset_Y))

# Cross-validation
for j, (train_idx, test_idx) in enumerate(folds):
    # Recuperar os dados de fold j
    print('\nFold ',j) # mostrar o Fold do treinamento
    X_train_cv = dataset_X[train_idx]
    Y_train_cv = dataset_Y[train_idx]
    X_test_cv = dataset_X[test_idx]
    Y_test_cv= dataset_Y[test_idx]

    # Autoencoder para testes provisionais (usando o mesmo conjunto de validacao e teste)
    Xtr = X_train_cv
    Ytr = Xtr
    Xval = X_test_cv
    Yval = Xval
    Y_test_cv = X_test_cv

    # Dividir o conjunto de Treinamento em Teste e Validacao
    #Xtr, Xval, Ytr, Yval = train_test_split(X_train_cv, Y_train_cv, stratify=Y_train_cv, test_size=0.20)

    # Etapa de entrenamento da rede
    [Yout_tr,vet_erro_tr,vet_erro_val,nit_parou] = oMLP.treinar_MLP(Xtr, Ytr, Xval, Yval, nitmax, alfa)

    # Etapa de teste da rede como autoencoder
    [Yout_test,EQM_test] = oMLP.testar_MLP(X_test_cv, Y_test_cv)
    print('\n EQM: ',EQM_test, 'nit_parou',nit_parou)

    # Calculo do erro do modelo quando se usou o conjunto de teste
    erro = Yout_test - Y_test_cv
    N=len(Yout_test)
    EQMs = np.sum(erro*erro,axis=1)/350 #/N

    Yd = pd.Series(dataset_Y[test_idx])
    Y = pd.Series(EQMs > 0.01)
    Y[EQMs>0.01] = 'a'
    Y[EQMs<=0.01] = 'n'

    Y.to_csv("Fold %s-Y.csv"%j, sep=',', encoding='utf-8', index=False)
    Yd.to_csv("Fold %s-Yd.csv"%j, sep=',', encoding='utf-8',index=False)

