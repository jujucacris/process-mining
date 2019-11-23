#!/usr/bin/python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import csv
import sys
import os
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve

def transpose_1(m):
    s = []
    for item in m:
        if item[0] == 'n':
            s.append(0)
        else:
            s.append(1)
    return s

def transpose_2(m):
    s = []
    for item in m:
        s.append(float(item[0]))
    return s

def setup():

    f = open(os.path.join("pos_processamento","entradas_roc.csv"),"r")
    f = csv.reader(f)
    f = next(f)
    n = f[2]
    y = open(os.path.join("pos_processamento","entradas",f[0]),"r")
    y = csv.reader(y)
    y = transpose_2(y)
    yd = open(os.path.join("pos_processamento","entradas",f[1]),"r")
    yd = csv.reader(yd)
    yd = transpose_1(yd)
    return y, yd, n

def roc(y, yd, f, j):
    fpr, tpr, thresholds = roc_curve(yd, y, pos_label=1)
    print(thresholds)

    fig = plt.figure(2)
    fig.set_size_inches(10,10)
    lw = 2
    plt.plot(fpr, tpr, lw=lw, label='Treinamento %s' % j)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    #plt.show()
    plt.savefig(os.path.join("resultados","curva_roc_%s.png" % f), bbox_inches='tight')


def precision(y, yd, f, j):
    precision, recall, thresholds = precision_recall_curve(yd, y, pos_label=1)
    print(thresholds)

    fig = plt.figure(3)
    fig.set_size_inches(10,10)
    lw = 2
    plt.plot(recall, precision, lw=lw, label='Treinamento %s' % j)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision X Recall curve')
    plt.legend(loc="lower right")
    #plt.show()
    plt.savefig(os.path.join("resultados","curva_precision_recall_%s.png" % f), bbox_inches='tight')



def main(arg, j):

    y = setup()
    n = y[2]
    yd = y[1]
    y = y[0]

    if arg == "r":
        roc(y,yd,n,j)
    elif arg == "p":
        precision(y,yd,n,j)

if __name__ == '__main__':

	main(sys.argv[1], sys.argv[2])
