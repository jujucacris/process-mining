#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 00:03:32 2019

@author: ubuntu
"""

from IPython import display
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import pandas as pd
import numpy as np
np.random.seed(0)
from matplotlib import pyplot as plt

dados = pd.DataFrame()
dados['x'] = np.linspace(-10,10,100)
dados['y'] = 5 + 3*dados['x'] + np.random.normal(0,3,100)

# define a função custo
def L(y, y_hat):
    return ((y-y_hat) ** 2).sum()

# define valores de b_hat e w_hat
b_hat, w_hat = np.linspace(-10,20,40), np.linspace(0,6,40)

# acha o custo para cada combinação de b_hat e w_hat
loss = np.array([L(dados['y'], i + j * dados['x']) for i in b_hat for j in w_hat]).reshape(40,40)
b_hat, w_hat = np.meshgrid(b_hat, w_hat) # combina os b_hat e w_hat em uma grade

# faz o gráfico em 3d
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_zticks([])
ax.set_xlabel('$\hat{b}$')
ax.set_ylabel('$\hat{w}$')
ax.set_zlabel('Custo', rotation=90)
surf = ax.plot_surface(b_hat, w_hat, loss,
                       rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0)
plt.show()

# implementa regressão linear com gradiente descendente
class linear_regr(object):

    def __init__(self, learning_rate=0.0001, training_iters=50):
        self.learning_rate = learning_rate
        self.training_iters = training_iters

    def fit(self, X_train, y_train):

        # formata os dados
        if len(X_train.values.shape) < 2:
            X = X_train.values.reshape(-1,1)
        X = np.insert(X, 0, 1, 1)

        # inicia os parâmetros com pequenos valores aleatórios
        # (nosso chute razoável)
        self.w_hat = np.random.normal(0,5, size = X[0].shape)

        for _ in range(self.training_iters):

            gradient = np.zeros(self.w_hat.shape) # inicia o gradiente

            # computa o gradiente com informação de todos os pontos
            for point, yi in zip(X, y_train):
                gradient +=  (point * self.w_hat - yi) * point

            # multiplica o gradiente pela taxa de aprendizado
            gradient *= self.learning_rate

            # atualiza os parâmetros
            self.w_hat -= gradient

    def predict(self, X_test):
        # formata os dados
        if len(X_test.values.shape) < 2:
            X = X_test.values.reshape(-1,1)
        X = np.insert(X, 0, 1, 1)

        return np.dot(X, self.w_hat)




    def _show_state(self, X_train, y_train, loss):
        # visualiza o processo de aprendizado
        lb = L(y_train, self.w_hat[0] + 3 * X_train) # calcula o custo na direção b
        lw = L(y_train, 5 + self.w_hat[1] * X_train) # calcula o custo na direção w

        # scatter plot
        plt.subplot(221)
        plt.scatter(X_train, y_train, s= 10)
        plt.plot(X_train, self.predict(X_train), c='r')
        plt.title('$y = b + w x$')
        plt.tick_params(labelsize=9, labelleft=False, labelbottom = False)
        plt.grid(True)

        # loss
        plt.subplot(222)
        plt.plot(range(len(loss)), loss)
        plt.title('Custo')
        plt.tick_params(labelsize=9, labelleft=False, labelbottom = False)
        plt.grid(True)

        # b_loss
        plt.subplot(223)
        plt.plot( np.linspace(-10,20,20), self.b_loss)
        plt.scatter(self.w_hat[0], lb, c = 'k')
        plt.title('Custo em $\hat{b}$')
        plt.tick_params(labelleft=False)
        plt.grid(True)

        # w_loss
        plt.subplot(224)
        plt.plot(np.linspace(0,6,20), self.w1_loss)
        plt.scatter(self.w_hat[1], lw, c = 'k')
        plt.title('Custo em $\hat{w}$')
        plt.grid(True)
        plt.tick_params(labelleft=False)

        plt.tight_layout()
        display.display(plt.gcf())
        display.clear_output(wait=True)
        plt.clf() # limpa a imagem do gráfico

regr = linear_regr(learning_rate=0.0005, training_iters=30)
regr.fit(dados['x'], dados['y'])