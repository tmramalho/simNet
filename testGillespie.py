'''
Created on Feb 17, 2012

@author: tiago
'''

import numpy as np
from stochSolver import stochSolver
from gillespieSolver import gill
import matplotlib.pyplot as plt

if __name__ == '__main__':
	A = np.array([[-0.1,0.8,-0.8],
				  [-0.8,-0.1,0.8],
				  [0.8,-0.8,-0.1]], dtype=float)
	
	g = gill(10)
	
	g.setInit(np.array([200,220,170]))
	points = g.solveLinearGillespie(A)
	lines = plt.plot(points[0], points[1])
	plt.setp(lines[0],color='r')
	plt.setp(lines[1],color='g')
	plt.setp(lines[2],color='b')
	
	ss = stochSolver(3, 10)
	ss.setInit(np.array([200,220,170]))
	x = ss.solveLinearDet(A, 0.01, 1000)
	lines = plt.plot(np.arange(0, 10, 0.01), x)
	ls = '-'
	a=0.9
	lw=2
	plt.setp(lines[0],color='r',ls=ls, alpha=a, linewidth=lw)
	plt.setp(lines[1],color='g',ls=ls, alpha=a, linewidth=lw)
	plt.setp(lines[2],color='b',ls=ls, alpha=a, linewidth=lw)
	
	c = ss.solveLinearCovMat(A, 0.01, 1000)
	d = np.sqrt(np.diagonal(c, axis1=1, axis2=2))
	ls = '--'
	a=0.8
	lw=2
	lines = plt.plot(np.arange(0, 10, 0.01), x+d)
	plt.setp(lines[0],color='r',ls=ls, alpha=a, linewidth=lw)
	plt.setp(lines[1],color='g',ls=ls, alpha=a, linewidth=lw)
	plt.setp(lines[2],color='b',ls=ls, alpha=a, linewidth=lw)
	lines = plt.plot(np.arange(0, 10, 0.01), x-d)
	plt.setp(lines[0],color='r',ls=ls, alpha=a, linewidth=lw)
	plt.setp(lines[1],color='g',ls=ls, alpha=a, linewidth=lw)
	plt.setp(lines[2],color='b',ls=ls, alpha=a, linewidth=lw)
	
	
	plt.show()