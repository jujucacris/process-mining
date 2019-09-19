import csv
import sys

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
    atividades_dict = dict(zip(atividades, range(1, i)))
    return atividades_dict

def converte(file, a):
    f = open(file, "r")
    f = csv.reader(f, delimiter=",")
    p = []
    r = open("%s-atividades.csv" % file.replace(".csv",""), "w")
    for atividade in a:
        p.append(atividade)
    p.sort()
    i = 0
    for row in f:
        if row[0] == str(i):
            atividade = a[row[2]] - 1
            p[atividade] = 1
        else:
            b = str(p).replace("[", "").replace("]", "\n")
            r.write(b)
            for n in range(len(a)):
                p[n] = 0



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
