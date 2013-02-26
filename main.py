'''
Created on Dec 25, 2011

@author: tiago
'''
import os
import numpy as np
from StatTest import test
from datetime import datetime

if __name__ == '__main__':
	workDir = os.getcwd() + '/data/' + datetime.now().strftime('%d%b%y_%H%M%S') + '/'
	os.mkdir(workDir)
	
	t = test(numRuns = 100, 
			time = 10, 
			dt = 0.001, 
			sigma = 0.4,
			dim = 3,
			wd = workDir)
	
	A = np.array([[0,6,0],
				  [3,0,0],
				  [0,2,0]], dtype=float)
	
	t.ss.setOscillatorParameters( A, np.array([4, 3, 4.5]), 
		0.5, np.array([4,0.4,0.8]), np.array([6, 5.4, 6.8]))
	
	t.oscSimulation(True)
	
	t.oscPrediction(True)