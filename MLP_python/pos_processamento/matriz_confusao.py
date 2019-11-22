#!/usr/bin/python
# -*- coding: utf-8 -*-

from sklearn.metrics import confusion_matrix
import csv
import os

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
def tabela(ver, pred, limiar, log, nome):

    print("Gerando matriz da saída '%s' sobre Yd '%s'" % (pred, ver))

    y_ver = open(os.path.join("pos_processamento","entradas",ver), "r")
    y_pred = open(os.path.join("pos_processamento","entradas",pred), "r")

    y_ver = csv.reader(y_ver, delimiter=",")
    y_pred = csv.reader(y_pred, delimiter=",")

    y_ver = transpose(y_ver)
    y_pred = transpose(y_pred)

    tn, fp, fn, tp = confusion_matrix(y_ver, y_pred).ravel()

    recall = round((tp/(tp+fn)),3)
    precision = round((tp/(tp+fp)),3)
    fscore = round(((2*(recall*precision))/(recall+precision)),3)

    return log, nome, float(limiar), tp, fp, tn, fn, fscore, recall, precision


def gera_matrizes():

    arquivos = open(os.path.join("pos_processamento","entradas.csv"), "r")
    arquivos = csv.reader(arquivos)
    line = next(arquivos)
    vetores = tabela(line[0], line[1], line[2], line[3], line[4])
    m = open(os.path.join("pos_processamento","matrizes_saida_%s.csv" % line[3]), "a")
    m.write(str(vetores).replace("(","").replace(")","")+"\n")
    m.close()


if __name__ == '__main__':

    gera_matrizes()
