#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plot

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#iris = datasets.load_iris()
#X = iris.data
#y = iris.target
#class_names = iris.target_names


def plot_confusion_matrix(nro_experimento,nro_iteracao,y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plot.cm.Blues,
                          ):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
            title_table = "normalizada"
        else:
            title = 'Confusion matrix, without normalization'
            title_table = "nao_normalizada"

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    classes = np.array(['a','n'])
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
        title_table = "normalizada"
    else:
        print('Confusion matrix, without normalization')
        title_table = "nao_normalizada"

    print(cm)
    fig = plot.figure((nro_experimento * 10) + 4)
    fig, ax = plot.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Rotulo real',
           xlabel='Rotulo predito')

    # Rotate the tick labels and set their alignment.
    plot.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plot.savefig(os.path.join("resultados","exp%s_matriz_confusao_%s_%s.png" % (nro_experimento,title_table,nro_iteracao)), bbox_inches='tight') # salvando a figura
    return ax




def mostrar_tabela_confusao_e_medidas_de_aval(nro_experimento,nro_iteracao):
    y_pred = pd.read_csv(os.path.join("pos_processamento","entradas","Exp%s_Iter%s_Y.csv" % (nro_experimento, nro_iteracao))).values
    y_true= pd.read_csv(os.path.join("pos_processamento","entradas","Exp%s_Iter%s_Yd.csv" % (nro_experimento, nro_iteracao))).values
    classes=np.array(['a','n'])

    # Plot non-normalized confusion matrix
    plot_confusion_matrix(nro_experimento,nro_iteracao,y_true, y_pred, classes=classes,title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plot_confusion_matrix(nro_experimento,nro_iteracao,y_true, y_pred, classes=classes, normalize=True,
                          title='Normalized confusion matrix')

    #Outras medidas
    print(classification_report(y_true, y_pred))


np.set_printoptions(precision=2)

#configurar parametros
#projeto_origem = os.getcwd()
#path_pasta=os.path.join(projeto_origem,"pos_processamento","entradas")
#nro_experimento=2
#nro_iteracao=0

#mostrar_tabela_confusao_e_medidas_de_aval(nro_experimento,nro_iteracao)
