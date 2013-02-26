'''
Created on Apr 30, 2012

@author: tiago
'''

import numpy as np
from bayesian import bayesian

if __name__ == '__main__':
	b = bayesian()
	data = []
	for i in range(0, 10):
		data.append([i, (np.random.rand(3)-0.5)*6])
		
	A = np.array([[0,6,0],
				  [3,0,0],
				  [0,2,0]], dtype=float)
	
	priorPar = np.concatenate((A.flatten(), np.array([4, 3, 4.5]), 
		np.array([1.4]), np.array([0,0,0]), np.array([0,0,0])))
	priorCov = np.eye(len(priorPar)) * 0.1
	
	b.mapEstimate(data, 0.01, 0.01, priorPar, priorCov)