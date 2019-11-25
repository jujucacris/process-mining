
# *******************************************
# DEFINICAO DOS EXPERIMENTOS
# v1.1
# *******************************************
from main_autoencoder import executar_autoencoder as executar_autoencoder
import pandas as pd
import os

# Experimento 1:
#nro_experimento = 2
#funcao_f = 'tan'  # funcao de ativacao da camada de entrada
#funcao_g = 'sig'  # funcao de ativacao da camada de saida
#nitmax = 2  # numero de iterações maximo(epocas)
#alfa = 0.8  # taxa de aprendizado
#no = 1  # numero de nos da camada oculta

#projeto_origem = 'D:\\GITHUB\\process-mining'

#k = 5 # iteracoes do crossvalidation



#Parametrizacao dos experimentos( Cada linha do dataframe sera um experimento)
experimentos=[[
        1, #nro_experimento
        'tan',#funcao_f # funcao de ativacao da camada de entrada
        'sig',#funcao_g  # funcao de ativacao da camada de saida
        2, #nitmax # numero de iterações maximo(epocas)
        0.8, #alfa  # taxa de aprendizado
        1, #no # numero de nos da camada oculta
        'p2p-0.3-1-usuarios-nolle.csv', #nome_dataset
        5, #k # iteracoes do crossvalidation
        -1# Inicializacao do EQMmean(Saida do experimento)
        ],[

        2, #nro_experimento
        'tan',#funcao_f # funcao de ativacao da camada de entrada
        'sig',#funcao_g  # funcao de ativacao da camada de saida
        10, #nitmax # numero de iterações maximo(epocas)
        0.8, #alfa  # taxa de aprendizado
        1, #no # numero de nos da camada oculta
        'p2p-0.3-1-usuarios-nolle.csv', #nome_dataset
        5, #k # iteracoes do crossvalidation
        -1# Inicializacao do EQMmean(Saida do experimento)
        ]
]
experimentos = pd.DataFrame(experimentos, columns=['nro_experimento','funcao_f','funcao_g','nitmax','alfa','no','nome_dataset','k_cv','EQMmean'])

# Execucao dos experimentos
for index, experimento in experimentos.iterrows():

    #Inicializando arquivo de matrices de confusao
    f = open(os.path.join("resultados","exp%s_matrizes_saida_%s.csv" % (experimento['nro_experimento'], experimento['nome_dataset'])),"w") # abrindo o arquivo de saídas de matrizes de confusão
    f.write("log,nome,limiar,TP,FP,TN,FN,F1-score,recall,precision\n") # escrevendo o cabeçalho
    f.close()

    #Executando nosso autoencoder
    EQMmean = executar_autoencoder(experimento['nro_experimento'],
                                   experimento['funcao_f'],
                                   experimento['funcao_g'],
                                   experimento['nitmax'],
                                   experimento['alfa'],
                                   experimento['no'],
                                   experimento['nome_dataset'],
                                   experimento['k_cv'])

    #Salvando resultados do autoencoder
    experimentos.loc[index,'EQMmean']=EQMmean
    #experimento['f-score']=EQMmean
    #experimento['precisao']=EQMmean
    #experimento['recall']=EQMmean


#Visualisando tabela resumo dos experimentos:

print('\n')
print("===============Tabela Resumo=================")
print(experimentos)

#Salvando tabela de experimentos ( parametros e resultados)
experimentos.to_csv(os.path.join("resultados",'experimentos_resumo.csv'), sep=',', encoding='utf-8',index=False)
