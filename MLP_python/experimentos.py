
# *******************************************
# DEFINICAO DOS EXPERIMENTOS
# v1.1
# *******************************************
from main_autoencoder import executar_autoencoder as executar_autoencoder
import pandas as pd
import os

#Parametrizacao dos experimentos( Cada linha do dataframe sera um experimento)
experimentos=[
#        [
#        14, #nro_experimento,
#        'teste cris 5',
#        'tan',#funcao_f # funcao de ativacao da camada de entrada
#        'sig',#funcao_g  # funcao de ativacao da camada de saida
#        2, #nitmax # numero de iterações maximo(epocas)
#        0.8, #alfa  # taxa de aprendizado
#        1, #no # numero de nos da camada oculta
#        'p2p-0.3-1-usuarios-nolle.csv', #nome_dataset
#        5 #k # iteracoes do crossvalidation
#        'autoencoder_proprio'
#        ],
        [
        20, #nro_experimento,
        'algoritmo nolle',
        '',#funcao_f # funcao de ativacao da camada de entrada
        '',#funcao_g  # funcao de ativacao da camada de saida
        50, #nitmax # numero de iterações maximo(epocas)
        0.8, #alfa  # taxa de aprendizado
        1148, #no # numero de nos da camada oculta
        'p2p-0.3-1-usuarios-nolle.csv', #nome_dataset
        5, #k # iteracoes do crossvalidation
        'autoencoder_nolle'
        ]
#,[
        #2, #nro_experimento,
        #'teste esther 1'
        #'tan',#funcao_f # funcao de ativacao da camada de entrada
        #'sig',#funcao_g  # funcao de ativacao da camada de saida
        #10, #nitmax # numero de iterações maximo(epocas)
        #0.8, #alfa  # taxa de aprendizado
        #1, #no # numero de nos da camada oculta
        #'p2p-0.3-1-usuarios-nolle.csv', #nome_dataset
        #5 #k # iteracoes do crossvalidation
        #],
        #[
        #3, #nro_experimento
        #'log curto'
        #'tan',#funcao_f # funcao de ativacao da camada de entrada
        #'sig',#funcao_g  # funcao de ativacao da camada de saida
        #10, #nitmax # numero de iterações maximo(epocas)
        #0.8, #alfa  # taxa de aprendizado
        #1, #no # numero de nos da camada oculta
        #'p2p-0.3-1-usuarios-curto.csv', #nome_dataset
        #5 #k # iteracoes do crossvalidation
        #]
#        4, #nro_experimento
#        '',
#        'tan',#funcao_f # funcao de ativacao da camada de entrada
#        'sig',#funcao_g  # funcao de ativacao da camada de saida
#        10, #nitmax # numero de iterações maximo(epocas)
#        0.8, #alfa  # taxa de aprendizado
#        10, #no # numero de nos da camada oculta
#        'p2p-0.3-1-usuarios-nolle.csv', #nome_dataset
#        5 #k # iteracoes do crossvalidation
#        ]

]
experimentos = pd.DataFrame(experimentos, columns=['nro_experimento','nome_experimento','funcao_f','funcao_g','nitmax','alfa','no','nome_dataset','k_cv','tipo_experimento'])
# Execucao dos experimentos
for index, experimento in experimentos.iterrows():

    #Inicializando arquivo de matrices de confusao
    f = open(os.path.join("resultados","exp%s_matrizes_saida_%s.csv" % (experimento['nro_experimento'], experimento['nome_dataset'])),"w") # abrindo o arquivo de saídas de matrizes de confusão
    f.write("log,nome,limiar,TP,FP,TN,FN,F1-score,recall,precision\n") # escrevendo o cabeçalho
    f.close()

    #Executando nosso autoencoder
    [EQMmean, limiar] = executar_autoencoder(experimento['nro_experimento'],
                                   experimento['funcao_f'],
                                   experimento['funcao_g'],
                                   experimento['nitmax'],
                                   experimento['alfa'],
                                   experimento['no'],
                                   experimento['nome_dataset'],
                                   experimento['k_cv'],
                                   experimento['tipo_experimento'])

    #Salvando resultados do autoencoder na variavel 'experimentos'
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
