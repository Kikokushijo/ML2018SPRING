import sys
test_case = sys.argv[2]
result = sys.argv[3]

with open(result, 'w+') as f:
	f.write('ID,Ans\n')
	for i in range(1980000):
		f.write('%d,0\n' % i)