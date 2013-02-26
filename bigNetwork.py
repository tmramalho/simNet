'''
Created on May 4, 2012

@author: tiago
'''

import os
import time
import numpy as np
import argparse
from datetime import datetime
from StatTest import test

if __name__ == '__main__':
	workDir = os.getcwd() + '/data/' + datetime.now().strftime('%d%b%y_%H%M%S') + '/'
	os.mkdir(workDir)
	paramFile = open(workDir + 'parameters.txt', 'w')
	#np.seterr('raise')
	
	parser = argparse.ArgumentParser(description='Simulate a stochastic network.')
	parser.add_argument('-ns', type=int, default=3,
					help='Number of species')
	parser.add_argument('-nr', type=int, default=100,
					help='Number of monte carlo runs')
	parser.add_argument('-t', type=int, default=10,
					help='Simulation time')
	parser.add_argument('-st', type=str, default='osc', 
					choices=['linear', 'osc', 'gene'],
					help='Simulation type')
	parser.add_argument('-sig', type=float, default=0.01,
					help='Noise strength')
	parser.add_argument('-pl', type=bool, default=False,
					help='Whether to plot monte carlo paths')
	parser.add_argument('-mu', type=float, default=0.5,
					help="Nonlinearity")
	
	args = parser.parse_args()
	
	t = test(numRuns = args.nr, 
			time = args.t, 
			dt = 0.001, 
			sigma = args.sig,
			dim = args.ns,
			wd = workDir,
			etype = args.st)
	
	if args.st == 'osc':
		if args.ns == 3:
			A = np.array([[0,2,0],[5,0,0],[0,3,0]])
		else:
			A = np.random.rand(args.ns,args.ns)*2+5
			for i in range(0,args.ns):
				for j in range(0,args.ns):
					if np.random.random_sample() < 0.6:
						A[i,j] = 0
			
		#w = np.random.rand(args.ns)+2
		mu = args.mu
		#x0 = np.random.rand(args.ns)*3 - 1.5
		#y0 = np.random.rand(args.ns)*3 - 1.5
		w  = np.array([2.0, 1.0, 2.0])
		x0 = np.array([2.0, 0, 0])
		y0 = np.array([0, 1.0, 2.0])
		
		
		paramFile.write( 'A:'  + str(A)  + '\n')
		paramFile.write( 'x0:' + str(x0) + '\n')
		paramFile.write( 'y0:' + str(y0) +  '\n')
		paramFile.write( 'w:'  + str(w)  +  '\n')
		paramFile.write( 'mu:' + str(mu) +  '\n')
		
		t.ss.setOscillatorParameters(A, w, mu, y0, x0)
		
	elif args.st == 'linear':
		if args.ns != 3:
			raise Exception 
		A = np.array([[-0.01,0.8,-0.8], [-0.8,-0.01,0.8], [0.8,-0.8,-0.01]], dtype=float)
		y0 = np.array([6, 5.4, 6.8])
		
		t.ss.setLinearParameters(A, y0)
		
		paramFile.write( str(A) + '\n' + str(y0) )
	elif args.st == 'gene':
		if args.ns != 3:
			raise Exception
		A = np.array([[0,  -1, 0], [0, 0,  -1], [ -1, 0, 0]], dtype=float)
		K = np.array([[0, 0.5, 0], [0, 0, 0.5], [0.5, 0, 0]], dtype=float)
		l = np.array([1,1,1])
		y0 = np.array([4, 0.01, 0.01])
		
		t.gs.setParams(2, A.T, K.T, l, y0, 10)
		
		paramFile.write( 'A:'  + str(A)  + '\n')
		paramFile.write( 'K:'  + str(K) + '\n')
		paramFile.write( 'y0:' + str(y0) +  '\n')
		paramFile.write( 'l:'  + str(l)  +  '\n')
		paramFile.write( 'coop:' + str(2) +  '\n')
		
	else:
		t.test()
		t.compareChiSquare()
		exit()
		
	paramFile.close()
		
	sTime = time.time()
	t.simpleSimulation(args.pl)
	eTime = time.time()
	print 'Simulation done in', int(eTime - sTime), 's'
	
	sTime = time.time()
	t.simplePrediction()
	eTime = time.time()
	print 'Prediction done in', int(eTime - sTime), 's'
	
	t.plotThings()
	t.compareChiSquare()
	t.klEvolution()
	

		