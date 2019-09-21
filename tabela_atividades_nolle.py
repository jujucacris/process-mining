import csv
import sys
import math

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
    atividades.sort()
    b = []
    for n in range(1, i):
        b.append(str(bin(n)).replace("0b", ""))
    atividades_dict = dict(zip(atividades, b))
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
            p.append(int(atividade))
        else:
            b = str(p).replace("[", "").replace("]", "\n")
            r.write(b)
            p = []
            i = i + 1

def main(file):

    atividades = descobre_atividades(file)
    converte(file, atividades)


if __name__ == '__main__':

    for arg in sys.argv:
        if arg == sys.argv[0]:
            continue
        else:
            main(arg)
