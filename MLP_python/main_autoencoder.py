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
    from pos_processamento.matriz_confusao import gera_matrizes
    from pos_processamento.curva_roc_sklearn import main as curva_roc
    #import matriz_confusao
    #import curva_roc_sklearn
    projeto_origem = os.getcwd() #"D:\\GITHUB\\process-mining"

    # definir a rede
    oMLP = cMLP(funcao_f, funcao_g, no)

    # ler conjunto de dados
    dataset = pd.read_csv(os.path.join(projeto_origem,"..","Conversor de JSON", nome_dataset))
    dataset_X = np.array(dataset.iloc[:, :-2]) #dados do dataset
    dataset_Y = np.array(dataset['n']) #rotulos do dataset

    # Divisao do conjunto de treinamento em kFold cada um com partes para Trainamento e Valid
    # Itercoes contém conjuntos estratificados para treinamento e teste
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

        # Etapa de treinamento da rede
        [Yout_tr, vet_erro_tr, vet_erro_val, nit_parou] = oMLP.treinar_MLP(Xtr, Xtr, Xval, Xval, nitmax, alfa)
        #np.savetxt("exp%s_iter%s_WA.csv"%(nro_experimento,j), oMLP.WA, delimiter=",")
        #np.savetxt("exp%s_iter%s_WB.csv"%(nro_experimento,j), oMLP.WB, delimiter=",")

        # Grafica de evolucao do EQM(funcao de perda) durante o treinamento
        grafica_evolucao_EQM(vet_erro_tr, vet_erro_val, nome_dataset, j,nro_experimento)

        # Etapa de teste da rede como autoencoder
        [Yout_test, EQM_test] = oMLP.testar_MLP(X_test, Y_test)
        iteracao_EQMs_nit.loc[j] = [j, EQM_test, nit_parou]
        print('\nIteracao: ', j, ' EQM_test: ', EQM_test, ' nit_parou(Treinamento): ', nit_parou)

        # Calculo dos erros de reproducao do modelo quando foi usado o conjunto de teste
        erro = Yout_test - Y_test
        N = len(Yout_test)
        ns = Yout_test.shape[1] #numero de saidas(numero de neuronios de saida)
        EQMs = np.sum(erro * erro, axis=1) / ns
        limiar = EQMs.mean()

        # Geracao das predicoes do modelo(Y)
        Y = pd.Series(EQMs > limiar)
        Yd = pd.DataFrame(Y_test_cv)  # rotulos
        Y[EQMs > limiar] = 'a'
        Y[EQMs <= limiar] = 'n'

        # Geracao arquivos para a matriz de confusao
        Y = pd.DataFrame(Y)
        EQMs = pd.DataFrame(EQMs)
        Xtrain = pd.DataFrame(X_train_cv)
        Xtest = pd.DataFrame(X_test_cv)

        EQMs.to_csv(os.path.join(projeto_origem,"pos_processamento","entradas","Exp%s_Iter%s_EQMs.csv" % (nro_experimento, j)), sep=',', encoding='utf-8', index=False)
        Y.to_csv(os.path.join(projeto_origem,"pos_processamento","entradas","Exp%s_Iter%s_Y.csv" % (nro_experimento, j)), sep=',', encoding='utf-8', index=False)
        Yd.to_csv(os.path.join(projeto_origem,"pos_processamento","entradas","Exp%s_Iter%s_Yd.csv" % (nro_experimento, j)), sep=',', encoding='utf-8', index=False)
        Xtrain.to_csv(os.path.join(projeto_origem,"pos_processamento","entradas","Exp%s_Iter%s_Xtrain.csv" % (nro_experimento, j)), sep=',', encoding='utf-8', index=False)
        Xtest.to_csv(os.path.join(projeto_origem,"pos_processamento","entradas","Exp%s_Iter%s_Xtest.csv" % (nro_experimento, j)), sep=',', encoding='utf-8', index=False)

        # Escrever no arquivo entradas.csv da Cris
        Yd_arquivo = "Exp%s_Iter%s_Yd.csv" % (nro_experimento, j) #arquivo no qual sera salvado o Yd
        Y_arquivo = "Exp%s_Iter%s_Y.csv" % (nro_experimento, j) #arquivo no qual sera salvado o Y

        sr = open(os.path.join("pos_processamento","entradas.csv"), "w+")
        line = "%s,%s,%s,%s,%s" % (Yd_arquivo,Y_arquivo, limiar, nome_dataset, nro_experimento)
        b = str(line)
        sr.write(b)
        sr.close()

        # Escrever no arquivo entradas_roc.csv da Cris
        EQM_arquivo="Exp%s_Iter%s_EQMs.csv" % (nro_experimento, j)

        sr = open(os.path.join("pos_processamento","entradas_roc.csv"), "w+")
        line = "%s,%s,%s" % (EQM_arquivo, Yd_arquivo, nome_dataset)
        b = str(line)
        sr.write(b)
        sr.close()

        # gerar matriz confusao
        gera_matrizes()

        #gerar curva roc
        curva_roc("r", j,nro_experimento)

        #gerar grafica precision recall
        curva_roc("p", j,nro_experimento)
        #call(["python", ".\\entradas\\matriz_confusao.py"])
        #call(["python", ".\\entradas\\curva_roc_sklearn.py"])
        #call(["python", os.path.join(projeto_origem,"Pos-processamento","matriz_confusao.py")])
        #call(["python", os.path.join(projeto_origem, "Pos-processamento", "curva_roc_sklearn.py")])
        break

    # Guardar resumo de iteracoes do cross validation
    iteracao_EQMs_nit.to_csv(os.path.join("resultados","Exp%s_EQMs_nit.csv"%nro_experimento), sep=',', encoding='utf-8', index=False)

    # retorna erro geral do modelo
    return iteracao_EQMs_nit["EQM"].mean()
