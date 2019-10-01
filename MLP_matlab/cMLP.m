classdef cMLP <handle
	properties 
		no;%no: Numero de neuronios na camada oculta
		WA;%pesos da conexao de neuronios da camada de entrada e  camada oculta
        WB;%pesos da conexao da neuronios da camada oculta e camada de saida
		funcao_f;%funcao de ativacao dos neuronios da camada oculta
		funcao_g;%funcao de ativacao dos neuronios da camada oculta
	end
	methods
		function this = cMLP(funcao_f,funcao_g,no)
            %funcao_f: posiveis valores: tan, sig,linear
            %funcao_g: posiveis valores: tan, sig,linear            
            this.no = no;
			this.funcao_f = funcao_f;
			this.funcao_g = funcao_g;
			this.WA = [];
			this.WB = [];
        end        
        
        function [Y,vet_erro,vet_erro_val,nit_melhor]=treinar_MLP(this,Xtr, Ytr,XVal, YVal,nitmax, alfa)%TODO add accuracy
            %parametros de entrada:
            %nitmax: Numero de iteracoes maximas( epocas)
            
            ne=size(Xtr, 2);%ne: numero de entradas
            [N,ns]=size(Ytr);%N: numero de instancias de treinamento % ns: Numero de saidas
            [N_val,~]=size(YVal);
            Xtr=[ones(N,1),Xtr];
            XVal=[ones(N_val,1),XVal];

            this.WA=rands(this.no,ne+1)/10;%mtrix pesos no x ne 
            this.WB=rands(ns,this.no+1)/10;%pesos
            
            [Y,Z]=this.calc_saida(Xtr,this.WA,this.WB,N,this.funcao_f,this.funcao_g);
            erro=Y-Ytr;

            EQM=sum(sum(erro.*erro))/N;%erro total
            nit=1;
            
            %Achar EQM em dados de validacao
            if(not(isempty(XVal)))
                YVal_out=this.calc_saida(XVal,this.WA,this.WB,N_val,this.funcao_f,this.funcao_g);
                erro_val=YVal_out-YVal;
                EQM_val=sum(sum(erro_val.*erro_val))/N_val;
                EQM_val_melhor=EQM_val; %a variable vai guardar o melhor EQM achado no conjunto de validacao
                WA_melhor=this.WA;% WA_melhor vai guardar os melhores WA
                WB_melhor=this.WB;% WA_melhor vai guardar os melhores WB
            end
            
            vet_erro=EQM;
            vet_erro_val=EQM_val;
            nit_val=0;
            while(EQM>=1e-6 && nit<nitmax && nit_val<10000)
                nit = nit+1;
                [gradA, gradB] = this.calc_grad(Xtr,Z,Y,erro,this.WB, N,this.funcao_f,this.funcao_g);
                dirA=-gradA;
                dirB=-gradB;                
                
                this.WB=this.WB-alfa*gradB;
                this.WA=this.WA-alfa*gradA;
                
                [Y,Z]=this.calc_saida(Xtr,this.WA,this.WB,N,this.funcao_f,this.funcao_g);
                erro = Y-Ytr;

                EQM = sum(sum(erro.*erro))/N;                
                vet_erro=[vet_erro;EQM];
                
                %validacao
                if(not(isempty(XVal)))
                    YVal_out=this.calc_saida(XVal,this.WA,this.WB,N_val,this.funcao_f,this.funcao_g);
                    erro_val=YVal_out-YVal;
                    EQM_val=sum(sum(erro_val.*erro_val))/N_val;
                    vet_erro_val=[vet_erro_val;EQM_val];
                    if( EQM_val <EQM_val_melhor)
                        nit_val=0;
                        EQM_val_melhor=EQM_val; %a variable vai guardar o melhor EQM achado no conjunto de validacao                        
                        WA_melhor=this.WA;% WA_melhor vai guardar os melhores WA
                        WB_melhor=this.WB;% WA_melhor vai guardar os melhores WB                                        
                        nit_melhor=nit;                                                                      
                    else
                        nit_val=nit_val+1;%nit_val é o numero de iteracoes nas quais o EQM_val vai aumentando
                    end
                end                
            end
            if(not(isempty(XVal)))
                this.WA=WA_melhor; %o objeto vai ficar com o melhor WA
                this.WB=WB_melhor; %o objeto vai ficar com o melhor WB
            end
        end

        function [Y,EQM]=testar_MLP(this,Xtest, Ytest) %Ana
            ne=size(Xtest,2);
            [N,ns]=size(Ytest);
            Xtest=[ones(N,1),Xtest];

            [Y,Z]=this.calc_saida(Xtest,this.WA,this.WB,N,this.funcao_f,this.funcao_g);
            erro = Y-Ytest;
            EQM = sum(sum(erro.*erro))/N;            
        end

        function [output]=ativacao(this,funcao,input) %Ana
            switch funcao
                case 'tan' %tangete hiperbolica
                    output=tanh(input); %fSaída entre 1 e -1
                case 'sig' %Sigmoide %Saída entre 0 e 1
                    output=1./(1+exp(-input));
                case 'linear'%Linear
                    output=input;            
            end
        end
        function d=derivada_funcao(this,funcao,input) %Ana
            switch funcao
                case 'sig'   
                    d=input.*(1-input);
                case 'tan'   
                    d=(1-(input.*input));
                case 'linear'   
                    d=1;
            end
        end

        function [Y,Z]=calc_saida(this,X,WA,WB,N,funcao_f,funcao_g) % Ana
            Zin=X*WA';
            Z=this.ativacao(funcao_f,Zin);

            Yin=[ones(N,1),Z]*WB';
            Y=this.ativacao(funcao_g,Yin);
        end

        function [grad_WA,grad_WB]=calc_grad(this,Xtr,Z,Y,erro,WB,N,funcao_f,funcao_g) % Ana
            df=this.derivada_funcao(funcao_f,Z);%calculo de derivadas das funcoes
            dg=this.derivada_funcao(funcao_g,Y);

            grad_WB = 1/N*(erro.*(dg))'*[ones(N,1),Z];
            dJdZ = (erro.*(dg))*WB(:,2:end);
            grad_WA = 1/N*(dJdZ.*(df))'*Xtr;
        end
    end
end
    