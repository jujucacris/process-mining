import pandas as pd
import numpy as np
from clase_MLP import cMLP as cMLP
from autoencoder_nolle_2018 import DAE as DAE
from graficas_autoencoder import grafica_evolucao_EQM as grafica_evolucao_EQM
# leer arquivo do log
dataset = pd.read_csv("datasets/p2p-0.3-1-usuarios-nolle.csv")
Xtest_with_labels=dataset.iloc[4001:4999,:]
dataset=dataset.iloc[:,0:2296]

funcao_f='sig'
funcao_g='sig'

#treinamento Xtr=np.array(dataset1.iloc[1:4000,0:2296]);
Xtr=np.array(dataset.iloc[1:4000,0:2296]);
Ytr=Xtr;

#test
Xtest=np.array(dataset.iloc[4001:4999,0:2296]);
Ytest=Xtest;
nitmax=100;
alfa=1;
no=40;

#Validacao
Xval=Xtest
Yval=Ytest

modelo="autoencoder_nolle"
#limiar1=0.5
limiar1=None
iteracao=1

if( modelo=="autoencoder_nolle"):
    params = dict(fator=.2,
                  no=None,
                  nitmax=3,
                  noise=None)        
    oDAE=DAE(params)
    oDAE.treinar(Xtr,Ytr,Xval,Yval)
    Yout_test=oDAE.test(Xtest,Ytest)

elif (modelo=="autoencoder_undercomplete"):
    oMLP = cMLP(funcao_f,funcao_g,no)
    [Yout_tr,vet_erro_tr,vet_erro_val,nit_parou]=oMLP.treinar_MLP(Xtr, Ytr,Xtest,Ytest,nitmax, alfa) #TODO add accuracy
    [Yout_test,EQM_test]=oMLP.testar_MLP(Xtest, Ytest)
    grafica_evolucao_EQM(vet_erro_tr,vet_erro_val)

Yd=Xtest_with_labels.iloc[:,2296]


if (limiar1 is not None) :# Se limiar do neuronio foi configurado
    #Calculando a saida da rede com limiar
    Yout_test_binario=Yout_test
    Yout_test_binario[Yout_test>limiar1]=1
    Yout_test_binario[Yout_test<=limiar1]=0
    
    #Calculando a classe: primeira forma
    Yout_test_binario==Ytest
    n_neuronios_saida=Xtr.shape[1]
    estado_reproducao=((Yout_test_binario==Ytest).sum(axis=1)==n_neuronios_saida) #True: reproduzido exatamente igual. False: reproduzidocom erros

    #Obtendo a classificacao do modelo( Predicao do modelo)
    Y=pd.Series(np.ones(estado_reproducao.shape))
    Y[estado_reproducao==False]='a'
    Y[estado_reproducao==True]='n'
    
    #Analise rapida: acuracia
    (Y.values==Yd.values).sum()
        
else :

    erro = Yout_test-Ytest
    N=len(Yout_test)
    EQMs = np.sum(erro*erro,axis=1)/2296
        
    #np.savetxt("EQM%s.csv"%(iteracao),EQMs , delimiter=",")
    
    limiar2=EQMs.mean()

    Y=pd.Series(EQMs>limiar2)
    Yd[EQMs>limiar2]
    Y[EQMs>limiar2]='a'
    Y[EQMs<=limiar2]='n'
    
     #Analise rapida: acuracia
    (Y.values==Yd.values).sum()
    
    #Y.to_csv("Y_iter%s_limiar%s.csv"%(iteracao,limiar2), sep=',', encoding='utf-8', index=False)
    #Yd.to_csv("Yd_iter%s_limiar%s.csv"%(iteracao,limiar2), sep=',', encoding='utf-8',index=False)