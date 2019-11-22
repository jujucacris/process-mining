#!/usr/bin/python
# -*- coding: utf-8 -*-

import csv

def corta_traces():

	f = open("p2p-0.3-1-nolle.csv", "r")
	f = csv.reader(f)
	#max = 14
	#size = 350
	chars = 225 # número máximo de caracteres de um trace com até 9 atividades

	csv_final = []
	i = 0
	for line in f:
		if "1" in line[225:]:
			i += 1
		else:
			row = line[:255]
			row.append(line[350])
			csv_final.append(row)
	print("popped %s items!" % i)

	s = open("p2p-0.3-1-curto.csv", "w+")
	for line in csv_final:
		s.write(str(line).replace("[","").replace("]","").replace("'","").replace(", ",",")+"\n")
	s.close()



if __name__ == '__main__':
	corta_traces()
