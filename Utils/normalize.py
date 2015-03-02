import numpy as np
import csv
import sys

def normalize(inputFile, outputFile, dim):

	f = open(inputFile,'r')

	x = {}
	for i in xrange(dim):
		x[i] = []

	for line in f:
		row = line.split(' ')
		for k in xrange(dim):
			x[k].append(float(row[k]))

	for i in xrange(len(x)):
		col = x[i]
		x[i] = []
		x_min = min(col)
		x_max = max(col)
		x_diff = x_max - x_min
		for j in xrange(len(col)):
			x[i].append(round(((col[j] - x_min)/x_diff) ,5))

	f = open(outputFile, 'w')
	print len(x[0])
	print len(x)
	for i in xrange(len(x[0])):
		for j in xrange(len(x)):
			f.write(str(x[j][i]).strip())
			f.write(' ')
		f.write('1')
		f.write('\n')
		
	f.close()
		
normalize(sys.argv[1], sys.argv[2], int(sys.argv[3]))
