#!/usr/bin/python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import csv
import sys
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

    f = open("entradas_roc.csv","r")
    f = csv.reader(f)
    f = next(f)
    y = open("./entradas/"+f[0])
    y = csv.reader(y)
    y = transpose_2(y)
    yd = open("./entradas/"+f[1])
    yd = csv.reader(yd)
    yd = transpose_1(yd)
    return y, yd

def roc(y, yd):
    fpr, tpr, thresholds = roc_curve(yd, y, pos_label=1)
    print(thresholds)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()

def precision(y, yd):
    precision, recall, thresholds = precision_recall_curve(yd, y, pos_label=1)
    print(thresholds)

    plt.figure()
    lw = 2
    plt.plot(recall, precision, color='darkblue',
             lw=lw, label='Precision x Recall curve')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision X Recall curve')
    plt.legend(loc="lower right")
    plt.show()


def main(arg):

    y = setup()
    yd = y[1]
    y = y[0]

    if arg == "r":
        roc(y,yd)
    elif arg == "p":
        precision(y,yd)

if __name__ == '__main__':

	main(sys.argv[1])
