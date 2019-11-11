
import pandas as pd
import numpy as np
from clase_MLP import cMLP as cMLP
# leer arquivo do log

dataset = pd.read_csv("datasets/p2p-0.3-1-nolle.csv")
Xtest_with_labels=dataset.iloc[4001:4999,:]
dataset=dataset.iloc[:,0:350]

dataset = pd.read_csv("Conversor de JSON/p2p-0.3-1-usuarios-nolle.csv")
Xtest_labels=dataset.iloc[4001:4999,:]
dataset=dataset.iloc[:,0:2296]


funcao_f='sig'
funcao_g='sig'

#treinamento
Xtr=np.array(dataset.iloc[1:4000,0:2296]);
Ytr=Xtr;

#test
Xtest=np.array(dataset.iloc[4001:4999,0:2296]);
Ytest=Xtest;
nitmax=10;
alfa=1;
no=40;



oMLP = cMLP(funcao_f,funcao_g,no)
[Yout_tr,vet_erro_tr,vet_erro_val,nit_parou]=oMLP.treinar_MLP(Xtr, Ytr,Xtest,Ytest,nitmax, alfa) #TODO add accuracy
[Yout_test,EQM_test]=oMLP.testar_MLP(Xtest, Ytest)

erro = Yout_test-Ytest
N=len(Yout_test)
<<<<<<< HEAD
EQMs = np.sum(erro*erro,axis=1)/350


Yd=Xtest_with_labels.iloc[:,350]
Y=pd.Series(EQMs>0.01)
Y[EQMs>0.01]='a'
Y[EQMs<=0.01]='n'


Y.to_csv("Y.csv", sep=',', encoding='utf-8', index=False)
Yd.to_csv("Yd.csv", sep=',', encoding='utf-8',index=False)
=======
EQMs = np.sum(erro*erro,axis=1)/2296
Xtest_labels[EQMs>0.036]
>>>>>>> 9c48617edd7370435b25ddf45f2faa15e26a9a56
