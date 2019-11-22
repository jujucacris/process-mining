
# *******************************************
# DEFINICAO DOS EXPERIMENTOS
# v1.1
# *******************************************
from main_autoencoder import executar_autoencoder as executar_autoencoder

# Experimento 1:
nro_experimento = 1
funcao_f = 'tan'  # funcao de ativacao da camada de entrada
funcao_g = 'sig'  # funcao de ativacao
nitmax = 10  # numero de iterações maximo
alfa = 0.8  # taxa de aprendizado
no = 10  # numero de nos da camada oculta
#projeto_origem = 'D:\\GITHUB\\process-mining'
nome_dataset = "p2p-0.3-1-nolle.csv"
k = 5 # iteracoes do crossvalidation


EQMmean = executar_autoencoder(nro_experimento, funcao_f, funcao_g, nitmax, alfa, no, nome_dataset, k)