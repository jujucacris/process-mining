import numpy as np
#Simple use of cMLP class
funcao_f='tan'
funcao_g='sig'

#Funcao AND
#Xtr=np.array([[1,0],[1,1],[0,0],[0,1]])
#Ytr=np.array([[0],[1],[0],[0]])

#Funcao XOR
Xtr=np.array([[1,0],[1,1],[0,0], [0,1]]) # conjunto de treinamento
Ytr=np.array([[1],[0],[0],[1]]) # labels do conjunto de treinamento

Xtest=Xtr # conjunto de teste
Ytest=Ytr # labels do conjunto de teste
nitmax=10000 # numero de iterações maximo
alfa=0.5 # taxa de aprendizado
no=5 # numero de nos da camada oculta

oMLP = cMLP(funcao_f,funcao_g,no)
[Yout_tr,vet_erro_tr,vet_erro_val,nit_parou]=oMLP.treinar_MLP(Xtr, Ytr,Xtest,Ytest,nitmax, alfa) #TODO add accuracy
[Yout_test,EQM_test]=oMLP.testar_MLP(Xtest, Ytest)

Xtr.shape
Ytr.shape
Xtest.shape
Ytest.shape