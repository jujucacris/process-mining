#*******************************************
# REDE NEURAL AUTOENCODER
# v1.0
#*******************************************

import numpy as np

class cMLP(object):
    def __init__(self,funcao_f,funcao_g,no):
        self.no = no # no: Numero de neuronios na camada oculta
        self.funcao_f = funcao_f # funcao de ativacao dos neuronios da camada oculta posiveis valores: tan, sig,linear
        self.funcao_g = funcao_g # funcao de ativacao dos neuronios da camada oculta posiveis valores: tan, sig,linear
        self.WA = [] # pesos da conexao de neuronios da camada de entrada e  camada oculta
        self.WB = [] # pesos da conexao da neuronios da camada oculta e camada de saida

    def treinar_MLP(self,Xtr, Ytr,XVal, YVal,nitmax, alfa):
        # Parametros de entrada
        # Xtr: conjunto de treinamento
        # Ytr: labels do conjunto de treinamento
        # XVal: conjunto de validacao
        # YVal: labels do conjunto de validacao
        # nitmax: numero de iterações maximo (epocas)
        # alfa: taxa de aprendizado

        vet_erro=[] #inicializamos o vetor de erros de treinamento
        vet_erro_val=[] #inicializamos o vetor de erros de validacao
        ne = Xtr.shape[1] # ne: numero de entradas
        [N,ns] = Ytr.shape # N: numero de instancias de treinamento, ns: Numero de saidas
        N_val = YVal.shape[0] # N_val: numero de instancias no conjunto de validacao
        Xtr = np.concatenate((np.ones((N,1),float),Xtr),axis=1) # agrega o bias ao conjunto treinamento
        XVal = np.concatenate((np.ones((N_val,1),float),XVal),axis=1) # agrega o bias ao conjunto de validacao

        # inicializa a matriz de pesos aleatoriamente
        self.WA= np.random.randn(self.no,ne+1)/10 # matriz de pesos camada de entrada
        self.WB= np.random.randn(ns,self.no+1)/10 # matriz de pesos camada oculta

        # calculo da saída para a primeira camada da rede
        [Y,Z]=self.calc_saida(Xtr,self.WA,self.WB,N,self.funcao_f,self.funcao_g) # Y: vetor saida da rede, Z: vetor saida da primeira camada
        erro=Y-Ytr #ekn = Yk(n) - Ydk(n)

        # calculo do erro quadratico medio
        EQM=sum(sum(erro*erro))/N

        # inicializacao do nro de epocas
        nit=1

        # calculo erro quadratico medio para os dados de validacao
        if(not(np.all(XVal==0))):
            [YVal_out, ZVal_out]=self.calc_saida(XVal,self.WA,self.WB,N_val,self.funcao_f,self.funcao_g)
            erro_val=YVal_out-YVal
            EQM_val=sum(sum(erro_val*erro_val))/N_val
            EQM_val_melhor=EQM_val  # guardar o melhor EQM achado no conjunto de validacao
            WA_melhor=self.WA # guardar os melhores pesos para a camada de entrada
            WB_melhor=self.WB # guardar os melhores pesos para a camada de saida

        vet_erro.append(EQM)
        vet_erro_val.append(EQM_val)
        nit_val=0
        while(EQM>=23 and nit<nitmax and nit_val<10):
            nit = nit+1
            [gradA, gradB] = self.calc_grad(Xtr,Z,Y,erro,self.WB, N,self.funcao_f,self.funcao_g)
            #dirA=-gradA
            #dirB=-gradB

            # atualizar os pesos
            self.WB=self.WB-alfa*gradB
            self.WA=self.WA-alfa*gradA

            # calculo do erro na epoca
            [Y,Z]=self.calc_saida(Xtr,self.WA,self.WB,N,self.funcao_f,self.funcao_g)
            erro = Y-Ytr

            EQM = sum(sum(erro*erro))/N
            vet_erro.append(EQM)

            # validacao
            if(not(np.all(XVal==0))): # si el conj val no es vacio
                [YVal_out,ZVal_out]=self.calc_saida(XVal,self.WA,self.WB,N_val,self.funcao_f,self.funcao_g)
                erro_val=YVal_out-YVal
                EQM_val=sum(sum(erro_val*erro_val))/N_val
                vet_erro_val.append(EQM_val)
                if( EQM_val < EQM_val_melhor):
                    nit_val=0
                    EQM_val_melhor=EQM_val  # guardar o melhor EQM achado no conjunto de validacao
                    WA_melhor=self.WA # guardar os melhores pesos para a camada de entrada
                    WB_melhor=self.WB # guardar os melhores pesos para a camada de saida
                    nit_melhor=nit
                else:
                    nit_val=nit_val+1 # guardar o numero de iteracoes nas quais o EQM_val vai aumentando
            if(not(np.all(XVal==0))):
                self.WA=WA_melhor  # manter sempre o melhor WA
                self.WB=WB_melhor  # manter sempre o melhor WB

        vet_erro=np.asarray(vet_erro)
        vet_erro_val=np.asarray(vet_erro_val)
        return [Y,vet_erro,vet_erro_val,nit_melhor]

    def testar_MLP(self,Xtest, Ytest):
        ne=Xtest.shape[1]
        [N,ns]=Ytest.shape
        Xtest=np.concatenate((np.ones((N,1),float),Xtest),axis=1)
        [Y,Z]=self.calc_saida(Xtest,self.WA,self.WB,N,self.funcao_f,self.funcao_g)
        erro = Y-Ytest
        EQM = sum(sum(erro*erro))/N
        return [Y,EQM]

    def ativacao(self,funcao,input):
        if funcao =='tan': # tangete hiperbolica
            output=np.tanh(input)  # fSaída entre 1 e -1
        elif funcao =='sig': # Sigmoide # Saída entre 0 e 1
            output=1/(1+np.exp(-input))
        elif funcao =='linear': # Linear/identidAD
            output=input
        return output

    # calcula a derivada de uma funcao para um valor
    def derivada_funcao(self, funcao,input):
        if funcao == 'sig':
            d=input*(1-input)
        elif funcao == 'tan':
            d=(1-(input*input))
        elif funcao == 'linear':
            d=1
        return d

    def calc_saida(self,X,WA,WB,N,funcao_f,funcao_g):
        Zin=X@(WA.T)
        Z=self.ativacao(funcao_f,Zin) # funcao para a primeira camada
        Yin=np.concatenate((np.ones((N,1),float),Z),axis=1)@(WB.T)
        Y=self.ativacao(funcao_g,Yin) # funcao de ativacao para a saida
        return [Y,Z]

    def calc_grad(self,Xtr,Z,Y,erro,WB,N,funcao_f,funcao_g):
        df=self.derivada_funcao(funcao_f,Z) # calculo de derivadas das funcoes
        dg=self.derivada_funcao(funcao_g,Y)
        grad_WB = 1/N*((erro*dg).T)@(np.concatenate((np.ones((N,1),float), Z),axis=1))
        dJdZ = (erro*dg)@WB[:,1:]
        grad_WA = 1/N*((dJdZ*df).T)@Xtr
        return grad_WA,grad_WB
