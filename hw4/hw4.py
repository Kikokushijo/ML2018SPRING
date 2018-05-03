import sys
import numpy as np

test_case = sys.argv[2]
result = sys.argv[3]

with open(test_case, 'r') as f:
	test = np.loadtxt(f, delimiter=',', skiprows=1)

with open(result, 'w+') as f:
	f.write('ID,Ans\n')
	for idx, x, y in test:
		f.write('%d,0\n' % idx)