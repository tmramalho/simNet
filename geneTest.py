'''
Created on Jan 7, 2013

@author: tiago
'''

import numpy as np
import matplotlib.pyplot as plt
from geneSolver import geneSolver
from geneFixedSolver import geneFixedSolver

if __name__ == '__main__':
	gs = geneSolver(3, 0.1)
	gsf = geneFixedSolver(3, 0.1)
	A = np.array([[0,  -1, 0], [0, 0,  -1], [ -1, 0, 0]], dtype=float)
	K = np.array([[0, 0.5, 0], [0, 0, 0.5], [0.5, 0, 0]], dtype=float)
	l = np.array([1,1,1])
	y0 = np.array([4, 0.01, 0.01])
	
	gs.setParams(2, A.T, K.T, l, y0, 10)
	gsf.setParams(2, A.T, K.T, l, y0, 10)
	
	testVal = np.array([-0.2, 0.6, 0.1])
	
	fs1 = gs.geneFunction(testVal)
	fs2 = gsf.geneFunction(testVal)
	
	js1 = gs.geneJacobian(testVal)
	js2 = gsf.geneJacobian(testVal)
	
	hs1 = gs.geneHessian(testVal)
	hs2 = gsf.geneHessian(testVal)
	
	print np.allclose(fs1, fs2)
	print np.allclose(js1, js2)
	print np.allclose(hs1, hs2)
	
	time = 10
	dt = 0.001
	ds = time/float(1000)
	steps = int(time / ds)
	
	(xd, cd) = gs.solveGeneticPred(time, dt, steps)
	dev = np.sqrt(np.diagonal(cd[:,-3:,-3:], axis1=1, axis2=2))
	t = np.arange(0, time, ds)
	plt.plot(t, xd)
	plt.savefig("geneAv.pdf")
	plt.clf()
	plt.plot(t, dev)
	plt.savefig("geneDev.pdf")
	plt.clf()
	print "AD version done"
	
	(xd, cd) = gsf.solveGeneticPred(time, dt, steps)
	dev = np.sqrt(np.diagonal(cd[:,-3:,-3:], axis1=1, axis2=2))
	t = np.arange(0, time, ds)
	plt.plot(t, xd)
	plt.savefig("geneFixedAv.pdf")
	plt.clf()
	plt.plot(t, dev)
	plt.savefig("geneFixedDev.pdf")
	plt.clf()
	print "hand version done"
	