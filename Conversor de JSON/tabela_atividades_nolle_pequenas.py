#!/usr/bin/python
# -*- coding: utf-8 -*-

import csv

def corta_traces():

	f = open("p2p-0.3-1-usuarios-nolle.csv", "r")
	f = csv.reader(f)
	#max = 14
	chars = 1476 # número máximo de caracteres de um trace com até 9 atividades
	trace = 2296

	csv_final = []
	i = 0
	for line in f:
		if "1" in line[chars:]:
			i += 1
			row = line
			row[2297] = "l"
			csv_final.append(row)
		else:
			row = line
			row[2297] = "c"
			csv_final.append(row)
	print("popped %s items!" % i)

	s = open("p2p-0.3-1-usuarios-curto-rotulos.csv", "w+")
	for line in csv_final:
		s.write(str(line).replace("[","").replace("]","").replace("'","").replace(", ",",")+"\n")
	s.close()



if __name__ == '__main__':
	corta_traces()
