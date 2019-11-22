#!/usr/bin/python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import csv


def main():
    m = open("matrizes_saida.csv", "r")
    m = csv.reader(m)
    next(m)
    x_fp = []
    y_tp = []
    for row in m:
        y_tp.append(round(float(row[8]), 2))
        x_fp.append(round((float(row[4]) / (float(row[5]) + float(row[4]))),2))
    plt.xlabel("FP rate")
    plt.ylabel("TP rate")
    plt.plot(x_fp,y_tp,"ro")
    plt.title('Receiver operating characteristic')
    plt.axis([0,1.05,0,1.05])
    plt.show()


if __name__ == '__main__':

    main()
