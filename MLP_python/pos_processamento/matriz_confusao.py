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
def tabela(ver, pred, limiar, log, nome): # Yd, Y, limiar utilizado, nome do log, nome do experimento

    print("Gerando matriz da saída '%s' sobre Yd '%s'" % (pred, ver))

    y_ver = open(os.path.join("pos_processamento","entradas",ver), "r") # abre o arquivo Yd
    y_pred = open(os.path.join("pos_processamento","entradas",pred), "r") # abre o arquivo Y

    y_ver = csv.reader(y_ver, delimiter=",")
    y_pred = csv.reader(y_pred, delimiter=",")

    y_ver = transpose(y_ver) # transpoe de 1 linha para 1 coluna
    y_pred = transpose(y_pred) # transpoe de 1 linha para 1 coluna

    tn, fp, fn, tp = confusion_matrix(y_ver, y_pred).ravel() # gera a matriz de confusão a partir das entradas

    recall = round((tp/(tp+fn)),3) # calculo de recall
    precision = round((tp/(tp+fp)),3) # calulo de precision
    fscore = round(((2*(recall*precision))/(recall+precision)),3) # calculo de fscore

    return log, nome, float(limiar), tp, fp, tn, fn, fscore, recall, precision # retorna nome do log, nome do experimento, limiar e demais medidas

# função para gerar as matrizes
def gera_matrizes():
    """
     Funcao que vai gerar as matrizes de confusao a partir do arquivo entradas.csv.
     O arquivo entradas.csv deve conter:
         Y : Predicoes do modelo
         Yd : Rotulos
         Limiar : atributo apenas descritivo
         nome do log : atributo apenas descritivo
         nome do experimento : atributo apenas descritivo
    """
    arquivos = open(os.path.join("pos_processamento","entradas.csv"), "r") # abre o arquivo de entrada
    arquivos = csv.reader(arquivos)
    line = next(arquivos) # seleciona a primeira linha
    vetores = tabela(line[0], line[1], line[2], line[3], line[4]) # armazena os restultados retornados pela função principal
    m = open(os.path.join("resultados","matrizes_saida_%s.csv" % line[3]), "a") # abre o arquivo para saída
    m.write(str(vetores).replace("(","").replace(")","")+"\n") # imprime os resultados no arquivo de saída
    m.close()


if __name__ == '__main__':

    gera_matrizes()
