# importar librerias
import pandas as pd

# leer arquivo do log

dataset = pd.read_csv("p2p-0.3-1-usuarios-nolle.csv")

#Conferindo tamanho da matrix de entradas
tamanho_maximo_trace=14
numero_atividades=25
numero_usuarios=139

nro_colunas_matrix_entrada=(numero_atividades+numero_usuarios)*tamanho_maximo_trace
print("colunas esperadas : %s"%  (nro_colunas_matrix_entrada)) 
print("colunas obtidas : %s"%  (dataset.shape[1]-2)) 



#Conferindo cross validation estratificado
porc_normais=(dataset['n']=='n').sum()/len(dataset)

porc_anomalias=(dataset['n']=='a').sum()/len(dataset)

print("% traces normais : %s"%  (porc_normais)) 
print("% tarces anomalos : %s"%  (porc_anomalias)) 

#Conferindo porcentagem de anomalis e normais em um fold( treinamento)
nro_anomalias_em_tr=(dataset.iloc[train_idx,2296]=='n').sum()
nro_normais_em_tr=(dataset.iloc[train_idx,2296]=='n').sum()
nro_em_tr=len(train_idx)
nro_anomalias_em_tr/nro_em_tr # 0.24437218609304653
nro_normais_em_tr/nro_em_tr # 0.7556278139069534