import pandas as pd
import numpy as np
from clase_MLP import cMLP as cMLP
from graficas_autoencoder import grafica_evolucao_EQM as grafica_evolucao_EQM
# leer arquivo do log
dataset = pd.read_csv("datasets/p2p-0.3-1-nolle.csv")
Xtest_with_labels=dataset.iloc[4001:4999,:]
dataset=dataset.iloc[:,0:350]

funcao_f='sig'
funcao_g='sig'

#treinamento
Xtr=np.array(dataset.iloc[1:4000,0:350]);
Ytr=Xtr;

#test
Xtest=np.array(dataset.iloc[4001:4999,0:350]);
Ytest=Xtest;
nitmax=10;
alfa=1;
no=40;


oMLP = cMLP(funcao_f,funcao_g,no)
[Yout_tr,vet_erro_tr,vet_erro_val,nit_parou]=oMLP.treinar_MLP(Xtr, Ytr,Xtest,Ytest,nitmax, alfa) #TODO add accuracy
[Yout_test,EQM_test]=oMLP.testar_MLP(Xtest, Ytest)
grafica_evolucao_EQM(vet_erro_tr,vet_erro_val)

erro = Yout_test-Ytest
N=len(Yout_test)
EQMs = np.sum(erro*erro,axis=1)/350


Yd=Xtest_with_labels.iloc[:,350]
Y=pd.Series(EQMs>0.01)
Y[EQMs>0.01]='a'
Y[EQMs<=0.01]='n'


Y.to_csv("Y.csv", sep=',', encoding='utf-8', index=False)
Yd.to_csv("Yd.csv", sep=',', encoding='utf-8',index=False)