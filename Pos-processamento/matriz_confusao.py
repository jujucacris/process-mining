#!/usr/bin/python
# -*- coding: utf-8 -*-

from sklearn.metrics import confusion_matrix
import csv

#função para transpor a entrada de 1 coluna para 1 linha
def transpose(m):
    s = []
    for item in m:
        if item[0] == 'a':
            s.append(0)
        else:
            s.append(1)
    return s

# função principal para gerar as matrizes de confusão
def tabela(ver, pred, limiar, nome):

    y_ver = open("./entradas/"+ver, "r")
    y_pred = open("./entradas/"+pred, "r")

    y_ver = csv.reader(y_ver, delimiter=",")
    y_pred = csv.reader(y_pred, delimiter=",")

    y_ver = transpose(y_ver)
    y_pred = transpose(y_pred)

    tn, fp, fn, tp = confusion_matrix(y_ver, y_pred).ravel()

    recall = round((tp/(tp+fn)),3)
    precision = round((tp/(tp+fp)),3)
    fscore = round(((2*(recall*precision))/(recall+precision)),3)

    return nome, float(limiar), tp, fp, tn, fn, fscore, recall, precision

def imprime(matrizes):
    m = open("matrizes_saida.csv", "w+")
    m.write("log,limiar,TP,FP,TN,FN,F1-score,recall,precision\n")
    for item in matrizes:
        m.write(str(item).replace("(","").replace(")","")+"\n")

def gera_matrizes():

    arquivos = open("entradas.csv", "r")
    arquivos = csv.reader(arquivos)
    matrizes = []
    for line in arquivos:
        matrizes.append(tabela(line[0], line[1], line[2], line[3]))
    imprime(matrizes)


if __name__ == '__main__':

    gera_matrizes()
