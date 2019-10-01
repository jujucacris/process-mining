function main_MLP_POO()
    clc;clear; close all;  
    normalize='minmax'; 
%     funcoes=["sig";"tan";"softmax"]
    funcao_f='tan';
    funcao_g='sig';
%     for i=[1:size(funcoes,1)]
        valores_series=[1]; %variacoes de numeros de series que desejamos avaliar. Posiveis valores :[1,2,3,4]
        valores_atrassos=[12:1:12]; %variacoes de numero de atrassos
        valores_no=[5:1:5]; %variacoes de numero de neuronios na camada oculta
        valores_alfa=[0.25:0.25:0.25];%posiveis valores de alfa fixa nos experimentos        
        executar_experimentos(valores_no,funcao_f,funcao_g,valores_series,valores_alfa,valores_atrassos,normalize)
%     end
    
end
function executar_experimentos(valores_no,funcao_f,funcao_g,valores_series,valores_alfa,valores_atrassos,normalize)    
    timestamp=datestr(now,'DDmmHHMMSSFFF');
    for num_serie=valores_series
        sumario={}; 
        vector_WA={};
        vector_WB={}; 
        nexp=0;        
        for no=valores_no
            for atrassos=valores_atrassos
                dataset_original=load(['datasets/serie',num2str(num_serie),'_trein.txt']);                     
                if(num_serie==1)
                    dataset_original=dataset_original(1:48);% Deletamos os ultimos dados da serie porque tem muito ruido
                elseif(num_serie==4)
                    dataset_original=dataset_original(1:261);% Deletamos os ultimos dados da serie porque tem muito ruido
                end     
                n_instancias=size(dataset_original,1);
                if(strcmp(normalize,'minmax'))
                    [dataset,mini,maxi]= normalizing('minmax',dataset_original);
                    writetable(array2table(dataset),['datasets/','serie',num2str(num_serie),'_normalized','.csv']);
                else
                    dataset=dataset_original;
                end                                 
                    [X,Y]=criar_atrassos(dataset,atrassos);
                    writetable(array2table(X),['datasets/','serie',num2str(num_serie),'_X_atrassos_',num2str(atrassos),'.txt'],'WriteVariableNames',false);
                    writetable(array2table(Y),['datasets/','serie',num2str(num_serie),'_Y_atrassos_',num2str(atrassos),'.txt'],'WriteVariableNames',false);
                    n_instancias_tr=round(size(X,1)*0.7); %0.70% para treinamento
                    n_instancias_test=size(X,1)-n_instancias_tr;                    
                    Xtr=X(1:n_instancias_tr,:);
                    Ytr=Y(1:n_instancias_tr,:);
                    Xtest=X(n_instancias_tr+1:end,:);
                    Ytest=Y(n_instancias_tr+1:end,:);                    
    
                fig=figure('visible','off');
                ax1=axes;

                for alfa=valores_alfa
                    nitmax=10000;    
                    nexp=nexp+1;
                    oMLP = cMLP(funcao_f,funcao_g,no);                    
                    [Yout_tr,vet_erro_tr,vet_erro_val,nit_parou]=oMLP.treinar_MLP(Xtr, Ytr,Xtest,Ytest,nitmax, alfa);%TODO add accuracy
                    [Yout_test,EQM_test]=oMLP.testar_MLP(Xtest, Ytest);
                    WA=oMLP.WA;
                    WB=oMLP.WB;
                    vector_WA(nexp,1)={size(WA)};
                    vector_WA(nexp,2)={WA};
                    vector_WB(nexp,1)={size(WB)};
                    vector_WB(nexp,2)={WB};
                    EQM_test_denorm=EQM_test;
                    %denormalizar
                    if(strcmp(normalize,'minmax'))
                        Yout_tr=denormalizing('minmax',Yout_tr,mini,maxi);
                        Yout_test=denormalizing('minmax',Yout_test,mini,maxi);
                        erro = Yout_test - denormalizing('minmax',Ytest,mini,maxi);
                        EQM_test_denorm = sum(sum(erro.*erro))/size(Ytest,1);
                    end
                    EQMtr_inicial=vet_erro_tr(1);%EQM inicial de treinamento
                    EQMtr_final=vet_erro_tr(end);%EQM final de treinamento

                    %plot 
                    fig2=figure('visible','off');
                    ax2=axes;                    
                    plot(ax2,1:n_instancias,dataset_original,'linewidth',2,'DisplayName','S.Original');
                    hold(ax2,'on');
                    plot(ax2,atrassos+1:atrassos+n_instancias_tr,Yout_tr,'linewidth',2,'DisplayName',['S.PreditaTreinam. (alfa=',num2str(alfa),')']);
                    plot(ax2,atrassos+n_instancias_tr+1:atrassos+n_instancias_tr+n_instancias_test,Yout_test,'linewidth',2,'DisplayName',['S.PreditaTest (alfa=',num2str(alfa),')']);
                    xlabel(ax2,'Tempo')
                    ylabel(ax2,'Valores')    
                    title(ax2,['Serie vs serie predita (','L:',num2str(atrassos),' h:',num2str(no),' norm:',normalize,')']);
                    hold(ax2,'off');
                    legend(ax2,'off');
                    legend(ax2,'show');
                    hold(ax1,'on');  
                    saveas(fig2,['resultados/','serie',num2str(num_serie),'_pVS','_norm',normalize,'_L',num2str(atrassos),'_h',num2str(no),'_f',funcao_f,'_g',funcao_g,'_alfa',num2str(alfa),'.png']);

                    %Todos os EQM da tabela 'sumario' sao os que a rede gera( se os dados nao foram normalizados, os EQM sserao dados nao normalizados). 
                    %So o EQM_test_real é o erro denormalizado
                    sumario(nexp,:)={normalize,no,atrassos,alfa,nit_parou,EQMtr_inicial,EQMtr_final,EQM_test,EQM_test_denorm,funcao_f,funcao_g};
                    %graficar EQM com variacoes de alfa
                    plot(ax1,1:size(vet_erro_tr,1),vet_erro_tr,'linewidth',2, 'DisplayName',['alfa=',num2str(alfa),' (trein)']);
                    plot(ax1,1:size(vet_erro_val),vet_erro_val,'linewidth',2, 'DisplayName',['alfa=',num2str(alfa),' (val)']);
                    legend(ax1,'off');
                    legend(ax1,'show');
                    hold(ax1,'on');        
                end
                xlabel(ax1,'Numero de epocas')
                ylabel(ax1,'EQM')
                title(ax1,['Evolução do EQM (','L:',num2str(atrassos),' h:',num2str(no),' norm:',normalize,')']);   

                hold(ax2,'off');
                saveas(fig,['resultados/','serie',num2str(num_serie),'_pEQM','_norm',normalize,'_L',num2str(atrassos),'_h',num2str(no),'_f',funcao_f,'_g',funcao_g,'.png']);
                T = cell2table(sumario);
                T.Properties.VariableNames = {'Normalized','no','L','alfa','nitparou','eqm_tr_i','eqm_tr_f','eqm_val','eqm_val_denorm','funcao_f','funcao_g'};        
                disp(T)
            end
        end

        writetable(T,['resultados/','serie',num2str(num_serie),'_sumario','_f',funcao_f,'_g',funcao_g,'_',timestamp,'.csv']);
        disp(['saved in','resultados/','serie',num2str(num_serie),'_sumario','_f',funcao_f,'_g',funcao_g,'_',timestamp,'.csv']);

        writetable(cell2table(vector_WA),['resultados/','serie',num2str(num_serie),'_vectorWA','_f',funcao_f,'_g',funcao_g,'_',timestamp,'.csv']);
        writetable(cell2table(vector_WB),['resultados/','serie',num2str(num_serie),'_vectorWB','_f',funcao_f,'_g',funcao_g,'_',timestamp,'.csv']);
    end
end
function [X,Y]=criar_atrassos(dataset,atrassos)
    i=1;
    X=[];
    Y=[];
    while(i<=size(dataset,1)-atrassos)    
       X(i,:)=dataset(i:i+atrassos-1);
       Y(i,1)=dataset(i+atrassos);
       i=i+1;
    end        
end
function [data,mini,maxi]= normalizing(name,data)
    if(strcmp(name,'minmax'))        
        maxi=max(data);
        mini=min(data);
        data=(data-mini)/(maxi-mini);%Desde0 até 
    end
end

function [data]= denormalizing(name,data,mini,maxi)
    if(strcmp(name,'minmax'))        
        data=data*(maxi-mini)+mini;%Desde0 até 
    end
end