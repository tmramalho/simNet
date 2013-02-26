'''
Created on Apr 30, 2012

@author: tiago
'''

import numpy as np
import scipy.optimize
from stochSolver import stochSolver
from extPlotter import extPlot

import adolc

class bayesian(object):
	'''
	performs bayesian inference
	'''


	def __init__(self):
		'''
		Constructor
		'''
		
	def mapEstimate(self, data, sysSigma, expSigma, priorParameters, priorCovariance):
		self.data = data
		[self.tpoints, self.dpoints] = zip(*data)
		self.tpoints = np.array(self.tpoints)
		self.dpoints = np.array(self.dpoints)
		self.dim = len(self.dpoints[0])
		self.numDataPoints = len(data)*self.dim
		self.dt = 0.01 #TODO make sure that dt fits the data
		self.tmax = self.tpoints.max()+10*self.dt
		
		if not np.shape(expSigma):
			self.es = np.eye(self.numDataPoints)*expSigma
		else:
			if np.shape(expSigma) != (self.numDataPoints, self.numDataPoints):
				raise Exception('Wrong dimensions')
			self.es = expSigma
		self.abar = priorParameters
		adim = len(priorParameters)
		if np.shape(priorCovariance) != (adim, adim):
			raise Exception('Wrong dimensions')
		self.acov = priorCovariance
		self.ss = stochSolver(self.dim, expSigma)
		
		'''adolc: tape a function's path'''
		ax = np.array([adolc.adouble(0) for n in range(adim)])
		adolc.trace_on(1)
		adolc.independent(ax)
		ay = self.hamiltonian(ax)
		adolc.dependent(ay)
		adolc.trace_off()
		
		xopt = scipy.optimize.fmin_bfgs(self.objectiveFunction, self.abar, 
									fprime=self.hamiltonianPrime, callback=self.progress)
		self.showResult(xopt)
		
	def objectiveFunction(self, a):
		if(np.any(a[:6] < 0)):
			return 1000
		else:
			return self.hamiltonian(a)
		
	def hamiltonian(self, a):
		self.ss.setPackedOscillatorParameters(a, dtype=a.dtype)
		
		(x,c) = self.ss.solveOscillatorDet(self.dt, int(self.tmax/self.dt))
		#self.debug(a, x, c)
		self.ddist = np.array(self.dpoints, dtype=a.dtype)
		
		for i in range(0, len(self.ddist)):
			pos = i/self.dt
			#print len(x), pos
			self.ddist[i] = self.ddist[i] - x[pos, 3:]
		
		paramDist = a - self.abar
		dataDist = self.ddist.flatten()
		
		pda = -0.5*np.dot(dataDist, np.dot(self.es, dataDist))
		pa = -0.5*np.dot(paramDist,np.dot(self.acov, paramDist))
		
		return pda + pa
	
	def hamiltonianPrime(self, a):
		return adolc.gradient(1, a)
	
	def progress(self, x):
		print x
	
	def showResult(self, a):
		self.ss.setPackedOscillatorParameters(a, dtype=a.dtype)
		
		(x,c) = self.ss.solveOscillatorDet(self.dt, int(self.tmax/self.dt))
		self.debug(a, x, c)
			
	def debug(self, a, x, c):
		self.run = extPlot()
		d = np.sqrt(np.diagonal(c[:,3:,3:], axis1=1, axis2=2))
		self.run.plotLines(x[:,3:], self.tmax, self.dt, ls='-', lw=2.0)
		self.run.plotLines(x[:,3:]+d, self.tmax, self.dt, ls='-')
		self.run.plotLines(x[:,3:]-d, self.tmax, self.dt, ls='-')
		self.run.plotLines(x[:,3:]-d, self.tmax, self.dt, ls='-')
		self.run.plotPoints(self.dpoints, self.tpoints)
		self.run.save('bayesDebug.pdf')