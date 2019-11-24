import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


#iris = datasets.load_iris()
#X = iris.data
#y = iris.target
#class_names = iris.target_names


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    classes = np.array(['a','n'])
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
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
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
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
    return ax


np.set_printoptions(precision=2)

projeto_origem = os.getcwd()
#dataset = pd.read_csv(os.path.join("pos_processamento","matrizes_saida_p2p-0.3-1-usuarios-nolle.csv"))
caminho=os.path.join(projeto_origem,"pos_processamento","entradas")
nro_experimento=1
j=0

y_pred = pd.read_csv(os.path.join(caminho,"Exp%s_Iter%s_Y.csv" % (nro_experimento, j))).values
y_true= pd.read_csv(os.path.join(caminho,"Exp%s_Iter%s_Yd.csv" % (nro_experimento, j))).values
classes=np.array(['a','n'])

# Plot non-normalized confusion matrix
plot_confusion_matrix(y_true, y_pred, classes=classes,title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(y_true, y_pred, classes=classes, normalize=True,
                      title='Normalized confusion matrix')

plt.show()
