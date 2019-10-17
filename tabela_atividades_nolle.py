#!/usr/bin/python
# -*- coding: utf-8 -*-

import csv
import sys
import math
import pprint

# função para gerar as representações binárias das atividades
def gera_numeros(atividades): # recebe o vetor com todas as atividades
    b = [] # vetor finalizado com o binário de todas as atividades
    s = [] # vetor binário para iteração
    i = 0 # iterador para contagem das atividades
    for item in atividades: # itera o vetor de atividades
        s.append("0") # adiciona um 0 no vetor binário para cada atividades
    for item in atividades: # itera o vetor de atividades
        s[i] = "1" # modifica o bit correspondente à atividade para 1
        b.append("".join(s)) # adiciona a representação da atividade ao vetor final
        s[i] = "0" # volta o bit para 0
        i = i + 1
    return b # retorna o vetor finalizado com os códigos binários

# função para descobrir todas as atividades presentes no log
def descobre_atividades(file): # recebe o nome do arquivo a ser tratado
    f = open(file, "r")
    f = csv.reader(f, delimiter=",") # abertura do arquivo csv como matriz
    s = open(("lista_atividades_" + file), "w")
    atividades = [] # vetor das atividades
    i = 1 # iterador
    next(f) # pula a primeira linha do arquivo
    for row in f: # itera as linhas do arquivo
        #line = next(f) # pula para a próxima linha (ignorando a primeira)
        if row[2] in atividades: # verifica se o nome da atividade já existe no vetor
            continue # se sim, passa para a próxima iteração
        else:
            atividades.append(row[2]) # se não, adiciona o nome da atividade no vetor
            i = i + 1
    c = gera_numeros(atividades) # armazena o vetor de representações binárias
    atividades.sort() # ordena as atividades alfabeticamente
    print(str(len(atividades)) + " atividades descobertas") # imprime a quantidade de atividades descobertas
    atividades_dict = dict(zip(atividades, c)) # cria um dicionário associando atividades e binários
    for item in atividades_dict:
        s.write(item + "\n")
    return atividades_dict # retorna o dicionário criado

# função para formatar as strings dos traces para impressão no arquivo
def formata(p):
    p = str(p).replace("[", "").replace("]", "\n")
    p = p.replace("'", "").replace(", ", "")
    p = ",".join(p)
    return p

# função para completar cada trace com '0's para que todos tenham o mesmo tamanho
def completa(rows, max_size): # recebe a matriz de linhas e o tamanho máximo dos traces desse log
    for row in rows: # itera as linhas a serem impressas
        tipo = row.pop() # remove o identificador de tipo do final da lista
        while len(row) < max_size: # enquanto a linha não tiver o mesmo tamanho do maior trace
            row.append("0000000000000000000000000") # completa a linha com atividades em branco
        row.append(tipo) # devolve o identificador de tipo ao final da lista
    return rows # retorna matriz de traces

# função principal da conversão de eventos para traces binários
def converte(file, a): # recebe o nome do arquivo de entrada e o dicionário de atividades
    f = open(file, "r")
    f = csv.reader(f, delimiter=",") # abre o csv como matriz
    p = [] # vetor onde será armazenado cada trace
    r = open("%s-nolle.csv" % file.replace(".csv",""), "w") # abre o arquivo de saída
    i = 1 # iterador de traces
    max_size = 0 # variável de tamanho máximo de trace
    rows = [] # matriz de saída que será impressa no arquivo
    next(f)
    for row in f: # para cada linha do arquivo de entrada
        #if i == 0: # se for a primeira iteração, pula para a segunda linha, evitando o cabeçalho
        #    i = i + 1
        #    continue
        if row[0] == str(i): # se o identificador do trace for igual ao que estava sendo tratado
            tipo = row[1] # armazena o tipo de evento, normal ou anomalia
            atividade = a[row[2]] # armazena a atividade realizada em representação binária a partir do dicionário
            p.append(atividade) # adiciona a atividade binária ao vetor do trace
        else: # caso o identificador do trace mude
            p.append(tipo[0]) # adiciona o tipo de trace ao final do vetor
            if len(p) > max_size: # se o comprimento do trace for maior do que o valor maior armazenado
                max_size = len(p) # subistitui o maior valor pelo comprimento do trace atual
            rows.append(p) # adiciona o trace à matriz final
            p = [] # zera o vetor do trace
            i = i + 1 # passa para o próximo identificador de trace
            atividade = a[row[2]] # armazena a nova atividade
            p.append(atividade) # adiciona atividade como primeira atividade do novo trace
    p.append(tipo[0])
    if len(p) > max_size:
        max_size = len(p)
    rows.append(p)
    rows = completa(rows, max_size) # chama função que nivela todos os traces
    for row in rows: # para cada linha na matriz de traces
        r.write(formata(row)) # imprime linha formatada no arquivo
    print(str(max_size) + " eventos no maior trace") # imprime o tamanho do maior trace
    print("done!\n")


def binarize(file):

    atividades = descobre_atividades(file)
    #pp = pprint.PrettyPrinter()
    #pp.pprint(atividades)
    converte(file, atividades)


if __name__ == '__main__':

    for arg in sys.argv:
        if arg == sys.argv[0]:
            continue
        else:
            binarize(arg)
