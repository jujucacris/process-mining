#!/usr/bin/python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import csv
import sys
import os
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve

# função para transposição do arquivo Yd
def transpose_1(m):
    s = [] # vetor final da entrada Yd
    for item in m: # para cada item no arquivo
        if item[0] == 'n': # caso o rótulo seja N
            s.append(0) # adicionar classe negativa no vetor Yd
        else: # caso o rótulo seja A
            s.append(1) # adicionar classe positiva no vetor Y
    return s # retorna vetor Yd


# função para transposição do arquivo Y
def transpose_2(m):
    s = [] # vetor final da entrada Y
    for item in m: # para cada item no arquivo Y
        s.append(float(item[0])) # adiciona classe no vetor de saída
    return s # retorna vetor Y


# função para leitura dos arquivos de entrada
def setup():

    f = open(os.path.join("pos_processamento","entradas_roc.csv"),"r") # arquivo com as entradas
    f = csv.reader(f)
    f = next(f) # leitura da primeira linha
    n = f[2] # armazenamento do nome do dataset
    y = open(os.path.join("pos_processamento","entradas",f[0]),"r") # arquivo Y
    y = csv.reader(y)
    y = transpose_2(y) # transposição da entrada Y
    yd = open(os.path.join("pos_processamento","entradas",f[1]),"r") # arquivo Yd
    yd = csv.reader(yd)
    yd = transpose_1(yd) # transposição do arquivo Yd
    return y, yd, n # retorna Y, Yd e nome do dataset


# função principal da curva ROC
def roc(y, yd, f, j,nro_experimento): # entradas: Y, Yd, nome do dataset, numero da iteração, numero do experimento
    fpr, tpr, thresholds = roc_curve(yd, y, pos_label=1) # aplicação da função roc_curve do sklearn
    #print('thresholds')
    #print(thresholds)

    fig = plt.figure((nro_experimento * 3) + 1) # configurando a figura a ser modificada
    fig.set_size_inches(10,10) # tamanho da figura
    lw = 2 # grossura da linha
    plt.plot(fpr, tpr, lw=lw, label='Treinamento %s' % j) # plotando as curva no gráfico
    plt.xlim([0.0, 1.0]) # limite eixo X
    plt.ylim([0.0, 1.05]) # limite eixo Y
    plt.xlabel('False Positive Rate') # label eixo X
    plt.ylabel('True Positive Rate') # label eixo Y
    plt.title('Receiver operating characteristic') # nome do gráfico
    plt.legend(loc="lower right") # posição da legenda
    #plt.show()
    plt.savefig(os.path.join("resultados","exp%s_curva_roc_%s.png" % (nro_experimento,f)), bbox_inches='tight') # salvando a figura


def precision(y, yd, f, j,nro_experimento): # entradas: Y, Yd, nome do dataset, numero da iteração, numero do experimentos
    precision, recall, thresholds = precision_recall_curve(yd, y, pos_label=1) # aplicação da função precision x recall do sklearn
    #print(thresholds)

    fig = plt.figure((nro_experimento * 3) + 2) # configurando a figura a ser modificada
    fig.set_size_inches(10,10) # tamanho da figura
    lw = 2 # grossura da linha
    plt.plot(recall, precision, lw=lw, label='Treinamento %s' % j) # plotando a curva no gráfico
    plt.xlim([0.0, 1.0]) # limite eixo X
    plt.ylim([0.0, 1.05]) # limite eixo Y
    plt.xlabel('Recall') # label eixo X
    plt.ylabel('Precision') # label eixo Y
    plt.title('Precision X Recall curve') # título do gráfico
    plt.legend(loc="lower right") # posição da legenda
    #plt.show()
    plt.savefig(os.path.join("resultados","exp%s_curva_precision_recall_%s.png" % (nro_experimento,f)), bbox_inches='tight') # salvando a figura

#função para adicionar o ponto do limiar do nolle às curvas
def add_nolle(nro_experimento, dataset):
    arquivo = open(os.path.join("resultados","exp%s_matrizes_saida_%s.csv" % (nro_experimento, dataset)),"r")
    arquivo = csv.reader(arquivo)
    next(arquivo)
    i = 0
    for line in arquivo:
        tp = int(line[3])
        fp = int(line[4])
        tn = int(line[5])
        fn = int(line[6])
        recall = float(line[8])
        precision = float(line[9])
        tp_rate = tp / (tp + fn)
        fp_rate = fp / (tn + fp)
        fig = plt.figure((nro_experimento * 3) + 1) # configurando a figura a ser modificada
        plt.plot(fp_rate, tp_rate, 'o', label='Nolle %s' % i)
        plt.savefig(os.path.join("resultados","exp%s_curva_roc_%s.png" % (nro_experimento,dataset)), bbox_inches='tight') # salvando a figura
        plt.legend(loc="lower right")
        fig = plt.figure((nro_experimento * 3) + 2) # configurando a figura a ser modificada
        plt.plot(recall, precision, 'o', label='Nolle %s' % i)
        plt.savefig(os.path.join("resultados","exp%s_curva_precision_recall_%s.png" % (nro_experimento,dataset)), bbox_inches='tight') # salvando a figura
        plt.legend(loc="lower right")
        i += 1

# função main chamada a partir do main_autoencoder
def main(arg, j,nro_experimento): # tipo do gráfico requisitado, número da iteração, número do experimento

    y = setup() # função para abrir os arquivos necessários, retorna y, yd e o nome do dataset
    n = y[2] # nome do dataset
    yd = y[1] # arquivo Yd
    y = y[0] # arquivo Y

    if arg == "r": # se foi solicitada uma curva roc
        roc(y,yd,n,j,nro_experimento) # chama curva roc, passa y, yd, nome do dataset, numero da iteração e numero do experimento
    elif arg == "p": # se for solicitada curva precision recall
        precision(y,yd,n,j,nro_experimento) # chama precision recall, passa y, yd, nome do dataset, numero da iteração e numero do experimento

if __name__ == '__main__':

	main(sys.argv[1], sys.argv[2])
