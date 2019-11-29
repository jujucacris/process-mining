#*******************************************
# DETECAO DE ANOMALIAS EM LOG DE EVENTOS
# USANDO AUTOENCODERS
# v1.0
#*******************************************

# Importacao de librarias
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

# Definicao da arquitetura da rede
funcao_f='tan' # funcao de ativacao da camada de entrada
funcao_g='sig' # funcao de ativacao
nitmax=10000 # numero de iterações maximo
alfa=0.8 # taxa de aprendizado
no=1148 # numero de nos da camada oculta

# Leitura do dataset
dataset_log = pd.read_csv("Conversor de JSON/p2p-0.3-1-usuarios-nolle.csv")

# Definicao dos conjuntos em Training e Test
dataset_log = dataset_log.iloc[:,:-1] # elimina ultima fila de "nan"
dataset_X = dataset_log.iloc[:,:-1] # separa os dados do rótulo
dataset_Y = dataset_log['n'] # rótulos do dataset
dataset_Y = pd.get_dummies(dataset_Y).loc[:,'a'] # convertir a binaria, classe possitiva = "anomaly"

Xtrain, Xtest, Ytrain, Ytest = train_test_split(dataset_X, dataset_Y, stratify=dataset_Y, test_size=0.25) # 75% para treino e 25% para teste
Xtrain, Xvalid, Ytrain, Yvalid = train_test_split(Xtrain, Ytrain, stratify=Ytrain, test_size=0.25)

# Convertir os dataframes em arrays
Xtrain = np.array(Xtrain)
Ytrain = Xtrain
Xvalid = np.array(Xvalid)
Yvalid = Xvalid
Xtest = np.array(Xtest)
Ytest = Xtest

# Treinar e testar o modelo
oMLP = cMLP(funcao_f,funcao_g,no)
[Yout_tr,vet_erro_tr,vet_erro_val,nit_parou]=oMLP.treinar_MLP(Xtrain, Ytrain, Xvalid, Yvalid, nitmax, alfa)
[Yout_test,EQM_test]=oMLP.testar_MLP(Xtest, Ytest)

# Usar cross_validation para treinar o modelo (a implementar)
#oMLP = cMLP(funcao_f,funcao_g,no)
#scores = cross_val_score(oMLP, dataset_X, dataset_Y, cv=10)
