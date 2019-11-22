# *******************************************
# VALIDACAO
# CROSS VALIDATION ESTRATIFICADO
# v1.2
# *******************************************
def executar_autoencoder(nro_experimento, funcao_f, funcao_g, nitmax, alfa, no, nome_dataset, k):
    # Importacao de librarias
    import os
    from subprocess import call
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split, StratifiedKFold
    from clase_MLP import cMLP as cMLP
    from graficas_autoencoder import grafica_evolucao_EQM as grafica_evolucao_EQM
    #import matriz_confusao
    #import curva_roc_sklearn
    projeto_origem = "D:\\GITHUB\\process-mining" #os.getcwd()

    # definir a rede
    oMLP = cMLP(funcao_f, funcao_g, no)

    # ler conjunto de dados
    dataset = pd.read_csv(os.path.join(projeto_origem, "Conversor de JSON", nome_dataset))
    dataset_X = np.array(dataset.iloc[:, :-2])
    dataset_Y = np.array(dataset['n'])

    # Divisao o conjunto de treinamento em kFold cada um com partes para Train e Valid
    # Itercoes contém conjuntos estratificados para trainamento e teste
    iteracao = list(StratifiedKFold(n_splits=k, shuffle=True).split(dataset_X, dataset_Y))
    iteracao_EQMs_nit = pd.DataFrame(columns=['iteracao','EQM','nit']) # matriz para almacenar os erros de cada iteracao

    # Cross-validation
    for j, (train_idx, test_idx) in enumerate(iteracao):
        # Recuperar os dados de fold j
        X_train_cv = np.array(dataset_X[train_idx])
        Y_train_cv = np.array(dataset_Y[train_idx])
        X_test_cv = np.array(dataset_X[test_idx])
        Y_test_cv = np.array(dataset_Y[test_idx])

        X_train = X_train_cv
        Y_train = X_train_cv
        X_test = X_test_cv
        Y_test = X_test_cv

        # Dividir o conjunto de Treinamento em Teste e Validacao
        Xtr, Xval, Ytr, Yval = train_test_split(X_train, Y_train, stratify=Y_train_cv, test_size=0.20)

        # Etapa de entrenamento da rede
        [Yout_tr, vet_erro_tr, vet_erro_val, nit_parou] = oMLP.treinar_MLP(Xtr, Xtr, Xval, Xval, nitmax, alfa)

        # Grafica de evolucao da rede
        grafica_evolucao_EQM(vet_erro_tr, vet_erro_val)

        # Etapa de teste da rede como autoencoder
        [Yout_test, EQM_test] = oMLP.testar_MLP(X_test, Y_test)
        iteracao_EQMs_nit.loc[j] = [j, EQM_test, nit_parou]
        print('\nIteracao: ', j, ' EQM: ', EQM_test, ' nit_parou: ', nit_parou)

        # Calculo do erro do modelo quando se usou o conjunto de teste
        erro = Yout_test - Y_test
        N = len(Yout_test)
        EQMs = np.sum(erro * erro, axis=1) / N
        limiar = np.sum(EQMs)/N
        # Geracao da matriz de confusao
        Y = pd.Series(EQMs > limiar)
        Yd = pd.DataFrame(Y_test_cv)  # rotulos
        Y[EQMs > limiar] = 'a'
        Y[EQMs <= limiar] = 'n'

        # Geracao arquivos para a matriz de confusao
        Y = pd.DataFrame(Y)
        EQMs = pd.DataFrame(EQMs)
        Xtrain = pd.DataFrame(X_train_cv)
        Xtest = pd.DataFrame(X_test_cv)
        EQMs.to_csv(os.path.join(projeto_origem, "Pos-processamento", "entradas", "Exp%s_Iter%s_EQMs.csv" % (nro_experimento, j)), sep=',', encoding='utf-8', index=False)
        Y.to_csv(os.path.join(projeto_origem,"Pos-processamento","entradas","Exp%s_Iter%s_Y.csv" % (nro_experimento, j)), sep=',', encoding='utf-8', index=False)
        Yd.to_csv(os.path.join(projeto_origem,"Pos-processamento","entradas","Exp%s_Iter%s_Yd.csv" % (nro_experimento, j)), sep=',', encoding='utf-8', index=False)
        Xtrain.to_csv(os.path.join(projeto_origem,"Pos-processamento","entradas","Exp%s_Iter%s_Xtrain.csv" % (nro_experimento, j)), sep=',', encoding='utf-8', index=False)
        Xtest.to_csv(os.path.join(projeto_origem,"Pos-processamento","entradas","Exp%s_Iter%s_Xtest.csv" % (nro_experimento, j)), sep=',', encoding='utf-8', index=False)

        # Escrever no arquivo entradas.csv da Cris
        sr = os.open(os.path.join(projeto_origem,"Pos-processamento","Exp%s_Iter%s_entradas.csv" % (nro_experimento, j)), os.O_RDWR | os.O_CREAT)
        line = "%s, %s, %s, %s, %s" % ("Exp%s_Iter%s_Y.csv" % (nro_experimento, j), "Exp%s_Iter%s_Yd.csv" % (nro_experimento, j), limiar, nome_dataset, nro_experimento)
        b = str.encode(line)
        os.write(sr, b)
        os.close(sr)

        # Escrever no arquivo entradas_roc.csv da Cris
        sr = os.open(os.path.join(projeto_origem, "Pos-processamento", "Exp%s_Iter%s_entradas_roc.csv" % (nro_experimento, j)), os.O_RDWR | os.O_CREAT)
        line = "%s, %s" % ("Exp%s_Iter%s_EQMs.csv" % (nro_experimento, j), "Exp%s_Iter%s_Yd.csv" % (nro_experimento, j))
        b = str.encode(line)
        os.write(sr, b)
        os.close(sr)

        # gerar matriz confusao
        call(["python", ".\\entradas\\matriz_confusao.py"])
        call(["python", ".\\entradas\\curva_roc_sklearn.py"])
        #call(["python", os.path.join(projeto_origem,"Pos-processamento","matriz_confusao.py")])
        #call(["python", os.path.join(projeto_origem, "Pos-processamento", "curva_roc_sklearn.py")])

    # Guardar resumo de iteracoes do cross validation
    iteracao_EQMs_nit.to_csv("iteraca_EQMs_nit.csv", sep=',', encoding='utf-8', index=False)

    # retorna erro geral do modelo
    return iteracao_EQMs_nit["EQM"].mean
