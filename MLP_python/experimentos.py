
# *******************************************
# DEFINICAO DOS EXPERIMENTOS
# v1.1
# *******************************************
def executar_experimentos(tipo_experimentos):
    from main_autoencoder import executar_autoencoder as executar_autoencoder
    import pandas as pd
    import os

    if(tipo_experimentos=='experimentos_outros'):
        #Parametrizacao dos experimentos( Cada linha do dataframe sera um experimento)
        experimentos=[
                [
                1, #nro_experimento,
                'teste esther 1',
                'tan',#funcao_f # funcao de ativacao da camada de entrada
                'sig',#funcao_g  # funcao de ativacao da camada de saida
                10, #nitmax # numero de iterações maximo(epocas)
                0.8, #alfa  # taxa de aprendizado
                10, #no # numero de nos da camada oculta
                'p2p-0.3-1-usuarios-nolle.csv', #nome_dataset
                5, #k # iteracoes do crossvalidation
                'autoencoder_proprio'        
                ]
        ]
        experimentos = pd.DataFrame(experimentos, columns=['nro_experimento','nome_experimento','funcao_f','funcao_g','nitmax','alfa','no','nome_dataset','k_cv','tipo_experimento'])                     
    elif(tipo_experimentos=='experimentos_traces_longos_curtos'):
        print('')
        #Parametrizacao dos experimentos( Cada linha do dataframe sera um experimento)
        #experimentos=[
        ###        [
        ###        14, #nro_experimento,
        ###        'teste cris 5',
        ###        'tan',#funcao_f # funcao de ativacao da camada de entrada
        ###        'sig',#funcao_g  # funcao de ativacao da camada de saida
        ###        2, #nitmax # numero de iterações maximo(epocas)
        ###        0.8, #alfa  # taxa de aprendizado
        ###        1, #no # numero de nos da camada oculta
        ###        'p2p-0.3-1-usuarios-nolle.csv', #nome_dataset
        ###        5 #k # iteracoes do crossvalidation
        ###        'autoencoder_proprio'
        ###        ],
        ##        [
        ##        20, #nro_experimento,
        ##        'algoritmo nolle',
        ##        '',#funcao_f # funcao de ativacao da camada de entrada
        ##        '',#funcao_g  # funcao de ativacao da camada de saida
        ##        50, #nitmax # numero de iterações maximo(epocas)
        ##        0.8, #alfa  # taxa de aprendizado
        ##        1148, #no # numero de nos da camada oculta
        ##        'p2p-0.3-1-usuarios-nolle.csv', #nome_dataset
        ##        5, #k # iteracoes do crossvalidation
        ##        'autoencoder_nolle'
        ##        ]
        #        [
        #        1, #nro_experimento,
        #        'teste esther 1',
        #        'tan',#funcao_f # funcao de ativacao da camada de entrada
        #        'sig',#funcao_g  # funcao de ativacao da camada de saida
        #        50, #nitmax # numero de iterações maximo(epocas)
        #        0.8, #alfa  # taxa de aprendizado
        #        10, #no # numero de nos da camada oculta
        #        'p2p-0.3-1-usuarios-nolle.csv', #nome_dataset
        #        5, #k # iteracoes do crossvalidation
        #        'autoencoder_proprio'        
        #        ]
        ##        #[
        ##        #3, #nro_experimento
        ##        #'log curto'
        ##        #'tan',#funcao_f # funcao de ativacao da camada de entrada
        ##        #'sig',#funcao_g  # funcao de ativacao da camada de saida
        ##        #10, #nitmax # numero de iterações maximo(epocas)
        ##        #0.8, #alfa  # taxa de aprendizado
        ##        #1, #no # numero de nos da camada oculta
        ##        #'p2p-0.3-1-usuarios-curto.csv', #nome_dataset
        ##        #5 #k # iteracoes do crossvalidation
        ##        #]
        ##
        #]
        #experimentos = pd.DataFrame(experimentos, columns=['nro_experimento','nome_experimento','funcao_f','funcao_g','nitmax','alfa','no','nome_dataset','k_cv','tipo_experimento'])             
    elif(tipo_experimentos=='autoencoder_nolle'): #Experimentos para achar os melhores parametros        
        print('')
    elif(tipo_experimentos=='experimentos_parametros'): #Experimentos para achar os melhores parametros
        #10 Experimentos basicos
        experimentos = pd.DataFrame({
            'nro_experimento':range(1,11),
            'nome_experimento':'experimentos basicos',
            'funcao_f':['sig','sig','sig','sig','sig','tan','tan','tan','tan','tan'],
            'funcao_g':'sig',
            'nitmax':1,
            'alfa':0.8,
            'no':[10,20,30,40,50,10,20,30,40,50],
            'nome_dataset':'p2p-0.3-1-usuarios-nolle.csv',
            'k_cv':5,
            'tipo_experimento':'autoencoder_proprio'
        })
        experimentos['nome_experimento']=experimentos['nome_experimento']+experimentos['nro_experimento'].astype(str)
            
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


tipo_experimentos='experimentos_outros'
executar_experimentos(tipo_experimentos)