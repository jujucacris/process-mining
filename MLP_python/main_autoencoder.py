# *******************************************
# VALIDACAO
# CROSS VALIDATION ESTRATIFICADO
# v1.1
# *******************************************

# Importacao de librarias
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from clase_MLP import cMLP as cMLP
from graficas_autoencoder import grafica_evolucao_EQM as grafica_evolucao_EQM

# Definicao da arquitetura da rede
funcao_f = 'tan'  # funcao de ativacao da camada de entrada
funcao_g = 'sig'  # funcao de ativacao
nitmax = 100  # numero de iterações maximo
alfa = 0.8  # taxa de aprendizado
no = 1148  # numero de nos da camada oculta
oMLP = cMLP(funcao_f, funcao_g, no)  # definir a rede

# ler conjunto de dados
dataset = pd.read_csv('/home/ubuntu/PycharmProjects/process-mining/Conversor de JSON/p2p-0.3-1-nolle.csv')
dataset_X = np.array(dataset.iloc[:, :-2])
dataset_Y = np.array(dataset['n'])

# definicao de parametros de treinamento
k = 5 # numero de folds para cross validation

# Divisao o conjunto de treinamento em kFold cada um com partes para Train e Valid
# Itercoes contém conjuntos estratificados para trainamento e teste
iteracao = list(StratifiedKFold(n_splits=k, shuffle=True).split(dataset_X, dataset_Y))
iteracao_EQMs_nit = pd.DataFrame(columns=['iteracao','EQM','nit']) # matriz para almacenar os erros de cada iteracao

# Cross-validation
for j, (train_idx, test_idx) in enumerate(iteracao):
    # Recuperar os dados de fold j
    X_train_cv = dataset_X[train_idx]
    Y_train_cv = dataset_Y[train_idx]
    X_test_cv = dataset_X[test_idx]
    Y_test_cv = dataset_Y[test_idx]

    # Dividir o conjunto de Treinamento em Teste e Validacao
    Xtr, Xval, Ytr, Yval = train_test_split(X_train_cv, Y_train_cv, stratify=Y_train_cv, test_size=0.20)

    # Etapa de entrenamento da rede
    [Yout_tr, vet_erro_tr, vet_erro_val, nit_parou] = oMLP.treinar_MLP(Xtr, Xtr, Xval, Xval, nitmax, alfa)

    # Grafica de evolucao da rede
    grafica_evolucao_EQM(vet_erro_tr, vet_erro_val)

    # Etapa de teste da rede como autoencoder
    [Yout_test, EQM_test] = oMLP.testar_MLP(X_test_cv, X_test_cv)
    iteracao_EQMs_nit.loc[j] = [j, EQM_test, nit_parou]
    print('\nIteracao: ', j, ' EQM: ', EQM_test, ' nit_parou: ', nit_parou)

    # Geracao arquivos para a matriz de confusao
    Y = pd.DataFrame(Yout_test)
    Yd = pd.DataFrame(dataset_Y[test_idx])
    Xtrain = pd.DataFrame(X_train_cv)
    Xtest = pd.DataFrame(X_test_cv)
    Y.to_csv("Y_iteracao_%s.csv" % j, sep=',', encoding='utf-8', index=False)
    Yd.to_csv("Yd_iteracao_%s.csv" % j, sep=',', encoding='utf-8', index=False)
    Xtrain.to_csv("Xtrain_iteracao_%s.csv" % j, sep=',', encoding='utf-8', index=False)
    Xtest.to_csv("Xtest_iteracao_%s.csv" % j, sep=',', encoding='utf-8', index=False)

    # Escrever no arquivo entradas.csv da Cris
    #("Y_iteracao_%s.csv", "Yd_iteracao_%s.csv", "limiarx", "p2p-0.3-1-nolle.csv", "experimento_1")


# Guardar resumo de iteracoes do cross validation
iteracao_EQMs_nit.to_csv("iteraca_EQMs_nit.csv", sep=',', encoding='utf-8', index=False)

# Erro geral do modelo
print(iteracao_EQMs_nit["EQM"].mean)
