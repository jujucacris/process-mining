import csv
import sys
import math
import pprint

def gera_numeros(atividades):
    b = []
    s = []
    i = 0
    for item in atividades:
        s.append("0")
    for item in atividades:
        s[i] = "1"
        b.append("".join(s))
        s[i] = "0"
        i = i + 1
    return b


def descobre_atividades(file):
    f = open(file, "r")
    f = csv.reader(f, delimiter=",")

    atividades = []
    i = 1
    for row in f:
        line = next(f)
        if line[2] in atividades:
            continue
        else:
            atividades.append(line[2])
            i = i + 1
    c = gera_numeros(atividades)
    #b = []
    #for n in range(1, i):
    #    b.append(str(bin(n)).replace("0b", ""))
    atividades.sort()
    atividades_dict = dict(zip(atividades, c))
    return atividades_dict

def converte(file, a):
    f = open(file, "r")
    f = csv.reader(f, delimiter=",")
    p = []
    r = open("%s-nolle.csv" % file.replace(".csv",""), "w")
    i = 0
    for row in f:
        if i == 0:
            i = i + 1
            continue
        if row[0] == str(i):
            atividade = a[row[2]]
            p.append(atividade)
            tipo = row[1]
        else:
            p.append(tipo[0])
            b = str(p).replace("[", "").replace("]", "\n")
            b = b.replace("'", "").replace(", ", "")
            b = ",".join(b)
            r.write(b)
            p = []
            i = i + 1

def main(file):

    atividades = descobre_atividades(file)
    pp = pprint.PrettyPrinter()
    pp.pprint(atividades)
    converte(file, atividades)


if __name__ == '__main__':

    for arg in sys.argv:
        if arg == sys.argv[0]:
            continue
        else:
            main(arg)
