'''
Created on May 31, 2012

@author: tiago
'''

from odeSolver import odeSolver
import pycppad as pcad
import numpy as np
import math

class geneSolver(object):
	'''
	classdocs
	'''


	def __init__(self, dim, sigma):
		'''
		Constructor
		'''
		self.dim = dim
		self.m = np.zeros(dim, dtype=float)
		self.N = np.eye(dim)*sigma
		self.rand = 0
		
	def setParams(self, coop, conn, K, dec, y0, ts):
		self.coop = coop
		self.conn = conn
		self.K = K
		self.decay = dec
		self.ts = ts
		self.dtype = y0.dtype
		self.y0 = np.log(y0) #convert to log concentration
		'''cppad setup'''
		ax = pcad.independent(np.zeros(self.dim))
		ay = self.geneFun(ax)
		self.gf = pcad.adfun(ax, ay)
		self.gf.optimize()
		
	def geneFun(self, x):
		y = pcad.ad(np.zeros(np.shape(x)))
		for i in range(self.dim):
			for j in range(self.dim):
				rateij = pcad.condexp_lt(pcad.ad(self.conn[i,j]), pcad.ad(0), 
							(np.exp( - x[i]) * np.power(self.K[i,j], self.coop)/
							(np.exp(x[j] * self.coop)+np.power(self.K[i,j], self.coop))),
							pcad.ad(0))
				y[i] += self.ts*rateij
		return y - self.decay
	
	def geneFunction(self, x):
		return self.gf.forward(0, x)
	
	def geneJacobian(self, x):
		return self.gf.jacobian(x)
	
	def geneHessian(self, x):
		l = self.dim
		res = np.zeros((l,l,l), dtype=self.dtype)
		idm = np.eye(l)
		for i in xrange(l):
			res[i,:,:] = self.gf.hessian(x, idm[i])
		return res
	
	def solveGeneticLangevin(self, time, dt, steps):
		'''Solve the langevin equation for the genetic model'''
		sdt = math.sqrt(dt)
		y = np.zeros((steps,self.dim))
		y[0] = self.y0 + self.rand*np.random.multivariate_normal(self.m,self.N)
		yn = np.copy(y[0])
		yp = np.copy(y[0])
		jt = int(time/float(steps*dt))
		for i in xrange(1, steps):
			for _ in xrange(0, jt):
				yp = np.copy(yn)
				xsi = np.random.multivariate_normal(self.m,self.N)
				yn = yp + dt*self.geneFunction(yp) + sdt*xsi
			y[i] = np.copy(yn)
		#return np.exp(y) #return actual concentration
		return y
	
	def odeFunction(self, x, c):
		h = self.geneHessian(x)
		fp = self.geneJacobian(x)
		f = self.geneFunction(x) + 0.5*np.tensordot(h, c, 2)
		g = np.dot(fp, c) + np.dot(c, fp.T) + self.N
		#print (f,g)
		
		return(f,g)
	
	def odelinearFunction(self, x, c):
		fp = self.geneJacobian(x)
		f = self.geneFunction(x)
		g = np.dot(fp, c) + np.dot(c, fp.T) + self.N
		
		return(f,g)
	
	def solveGeneticPred(self, time, dt, steps):
		'''Predict the average value for the coupled gene'''
		x0 = self.y0
		'''inverse covariance matrix'''
		if self.rand:
			c0 = self.N
		else:
			c0 = np.zeros(self.dim, dtype=self.dtype)
		
		od = odeSolver()
		
		(x,c) = od.adamsSolver(time, dt, steps, self.dim, self.dtype, 
							self.odeFunction, x0, c0)
		
		'''lx = np.zeros((steps,self.dim), dtype=self.dtype)
		lc = np.zeros((steps, self.dim, self.dim), dtype=self.dtype)
		for i in xrange(0, steps):
			lx[i] = np.exp(x[i] + 0.5*np.diag(c[i]))
			lc[i] = np.exp(np.add.outer(x[i], x[i]))*(np.exp(2*c[i]) - 
							np.exp(0.5*np.add.outer(np.diag(c[i]),np.diag(c[i]))))
			
		return(lx, lc)'''
		return (x,c)
	
	def solveGeneticDet(self, time, dt, steps):
		'''Predict the average value for the coupled gene'''
		x0 = self.y0
		'''inverse covariance matrix'''
		if self.rand:
			c0 = self.N
		else:
			c0 = np.zeros(self.dim, dtype=self.dtype)
		
		od = odeSolver()
		
		(x,c) = od.adamsSolver(time, dt, steps, self.dim, self.dtype, 
							self.odelinearFunction, x0, c0)
		
		'''lx = np.zeros((steps,self.dim), dtype=self.dtype)
		lc = np.zeros((steps, self.dim, self.dim), dtype=self.dtype)
		for i in xrange(0, steps):
			lx[i] = np.exp(x[i] + 0.5*np.diag(c[i]))
			lc[i] = np.exp(np.add.outer(x[i], x[i]))*(np.exp(2*c[i]) - 
							np.exp(0.5*np.add.outer(np.diag(c[i]),np.diag(c[i]))))
			
		return(lx, lc)'''
		return (x,c)
	