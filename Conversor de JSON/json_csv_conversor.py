#!/usr/bin/python
# -*- coding: utf-8 -*-

import json
import sys
from tabela_atividades_nolle import binarize

# função que escreve o csv correspondente ao json
def write_csv(f, filename):
    filename = filename.replace(".json", "-usuarios.csv") # abre o arquivo de saída
    c = open(filename, "w+")
    c.write("traceid,label,activity,user\n") # escreve o cabeçalho do csv
    i = 1 # iterador para os ids de traces
    user_anomalies = [] # vetor de todos os ids de anomalias de usuário que terão seus rótulos modificados
    users = [] # vetor de todos os usuários encontrados
    for case in f['cases']: # iteração por cada case
        for event in case['events']: # iteração por cada evento dentro do trace
            if event['attributes']['user'] not in users: # se for um usuário ainda não identificado
                users.append(event['attributes']['user']) # armazene este usuário
            if case['attributes']['label'] != "normal": # se não for um trace normal
                #if case['attributes']['label']['anomaly'] == "Attribute": # se for uma anomalia de atributo (usuário)
                    #if case['id'] not in user_anomalies: # se o id do case ainda não estiver no array de anomalias de usuário
                        #user_anomalies.append(case['id']) # armazenar o id do trace
                    #label = "normal" # escrever a label desse trace como normal
                #else: # caso não seja uma anomalia de usuário
                    #label = "anomaly" # escrever a label desse trace como anomalia
                label = "anomaly"
            else: # caso o trace não seja uma anomalia
                label = "normal" # escrever a label do trace como normal
            c.write("%d,%s,%s,%s\n" % (i, label, event['name'], event['attributes']['user'])) # escrever no arquivo as informações pertinentes do evento
        i = i + 1 # aumentar o iterador de case
    print(str(len(user_anomalies)) + " anomalias de usuário tratadas")
    print(str(len(users)) + " usuários diferentes")

def main():
    e = open("entradas.txt", "r") # abrir o arquivo de nomes de arquivos de entradas
    for file in e: # para cada nome de arquivo
        file = file.replace("\n", "") # formata o nome do arquivo
        print(file)
        f = open(file, "r", encoding="utf-8")
        f = json.load(f) # abrir o json utilizando a biblioteca apropriada
        write_csv(f, file) # escrever a primeira conversão para um csv de eventos
        #binarize(file.replace("json", "csv")) # escrever a segunda conversão dos eventos para a representação binária do nolle


if __name__ == "__main__":

    main()
