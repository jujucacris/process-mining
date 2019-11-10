clc;clear; close all;
    
%Simple use of cMLP class
funcao_f='sig';
funcao_g='sig';

%Funcao AND
dataset=csvread('datasets/p2p_2.csv');    
Xtr=dataset(1:4000,:);
Ytr=Xtr;

%Funcao XOR
% Xtr=[1 0;1 1;0 0; 0 1];
% Ytr=[1;0;0; 1];

Xtest=dataset(4001:4999,:);
Ytest=Xtest;
nitmax=10; 
alfa=1;
no=20;


oMLP = cMLP(funcao_f,funcao_g,no);                    
[Yout_tr,vet_erro_tr,vet_erro_val,nit_parou]=oMLP.treinar_MLP(Xtr, Ytr,Xtest,Ytest,nitmax, alfa);%TODO add accuracy
[Yout_test,EQM_test]=oMLP.testar_MLP(Xtest, Ytest);

fig=figure('visible','on');
ax1=axes;
plot(ax1,1:size(vet_erro_tr,1),vet_erro_tr,'linewidth',2, 'DisplayName',['alfa=',num2str(alfa),' (trein)']);
hold(ax1,'on');
plot(ax1,1:size(vet_erro_val),vet_erro_val,'linewidth',2, 'DisplayName',['alfa=',num2str(alfa),' (val)']);
legend(ax1,'off');
legend(ax1,'show');
hold(ax1,'on');        
xlabel(ax1,'Numero de epocas')
ylabel(ax1,'EQM')
title(ax1,['Evolução do EQM']);   

