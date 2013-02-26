'''
Created on Jan 7, 2013

@author: tiago
'''

from odeSolver import odeSolver
import numpy as np
import math

class geneFixedSolver(object):
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
		
	def geneFun(self, x):
		y = np.zeros(3)
		y[0] = (np.exp( - x[0]) * np.power(self.K[0,2], self.coop)/
				(np.exp(x[2] * self.coop)+np.power(self.K[0,2], self.coop)))
		y[1] = (np.exp( - x[1]) * np.power(self.K[1,0], self.coop)/
				(np.exp(x[0] * self.coop)+np.power(self.K[1,0], self.coop)))
		y[2] = (np.exp( - x[2]) * np.power(self.K[2,1], self.coop)/
				(np.exp(x[1] * self.coop)+np.power(self.K[2,1], self.coop)))
		return self.ts*y - self.decay
	
	def geneFunction(self, x):
		return self.geneFun(x)
	
	def geneJacobian(self, val):
		K1 = self.K[0,2]
		K2 = self.K[1,0]
		K3 = self.K[2,1]
		x = val[0]
		y = val[1]
		z = val[2]
		res = [[-(K1**self.coop/(np.exp(x)*(np.exp(self.coop*z) + K1**self.coop))), 0, 
  -((np.exp(-x + self.coop*z)*K1**self.coop*self.coop)/(np.exp(self.coop*z) + K1**self.coop)**2)], 
 [-((np.exp(self.coop*x - y)*K2**self.coop*self.coop)/(np.exp(self.coop*x) + K2**self.coop)**2), -(K2**self.coop/(np.exp(y)*(np.exp(self.coop*x) + K2**self.coop))), 
  0], [0, -((np.exp(self.coop*y - z)*K3**self.coop*self.coop)/(np.exp(self.coop*y) + K3**self.coop)**2), 
  -(K3**self.coop/(np.exp(z)*(np.exp(self.coop*y) + K3**self.coop)))]]
		return self.ts*np.array(res)
	
	def geneHessian(self, val):
		K1 = self.K[0,2]
		K2 = self.K[1,0]
		K3 = self.K[2,1]
		x = val[0]
		y = val[1]
		z = val[2]
		n = self.coop
		res = [[[(np.exp(x) + np.exp(x + n*z)/K1**n)**(-1), 0, (np.exp(-x + n*z)*K1**n*n)/(np.exp(n*z) + K1**n)**2], 
  [0, 0, 0], [(np.exp(-x + n*z)*K1**n*n)/(np.exp(n*z) + K1**n)**2, 0, 
   (np.exp(-x + n*z)*K1**n*(np.exp(n*z) - K1**n)*n**2)/(np.exp(n*z) + K1**n)**3]], 
 [[(np.exp(n*x - y)*K2**n*(np.exp(n*x) - K2**n)*n**2)/(np.exp(n*x) + K2**n)**3, 
   (np.exp(n*x - y)*K2**n*n)/(np.exp(n*x) + K2**n)**2, 0], 
  [(np.exp(n*x - y)*K2**n*n)/(np.exp(n*x) + K2**n)**2, (np.exp(y) + np.exp(n*x + y)/K2**n)**(-1), 0], 
  [0, 0, 0]], [[0, 0, 0], [0, (np.exp(n*y - z)*K3**n*(np.exp(n*y) - K3**n)*n**2)/
    (np.exp(n*y) + K3**n)**3, (np.exp(n*y - z)*K3**n*n)/(np.exp(n*y) + K3**n)**2], 
  [0, (np.exp(n*y - z)*K3**n*n)/(np.exp(n*y) + K3**n)**2, (np.exp(z) + np.exp(n*y + z)/K3**n)**(-1)]]]
		return self.ts*np.array(res)
	
	def geneMango(self, val):
		K1 = self.K[0,2]
		K2 = self.K[1,0]
		K3 = self.K[2,1]
		x = val[0]
		y = val[1]
		z = val[2]
		n = self.coop
		res = [[[[-(K1**n/(np.exp(x)*(np.exp(n*z) + K1**n))), 0, 
    -((np.exp(-x + n*z)*K1**n*n)/(np.exp(n*z) + K1**n)**2)], [0, 0, 0], 
   [-((np.exp(-x + n*z)*K1**n*n)/(np.exp(n*z) + K1**n)**2), 0, 
    (np.exp(-x + n*z)*K1**n*(-np.exp(n*z) + K1**n)*n**2)/(np.exp(n*z) + K1**n)**3]], 
  [[0, 0, 0], [0, 0, 0], [0, 0, 0]], 
  [[-((np.exp(-x + n*z)*K1**n*n)/(np.exp(n*z) + K1**n)**2), 0, 
    (np.exp(-x + n*z)*K1**n*(-np.exp(n*z) + K1**n)*n**2)/(np.exp(n*z) + K1**n)**3], [0, 0, 0], 
   [(np.exp(-x + n*z)*K1**n*(-np.exp(n*z) + K1**n)*n**2)/(np.exp(n*z) + K1**n)**3, 0, 
    -((np.exp(-x + n*z)*K1**n*(np.exp(2*n*z) - 4*np.exp(n*z)*K1**n + K1**(2*n))*n**3)/
      (np.exp(n*z) + K1**n)**4)]]], 
 [[[-((np.exp(n*x - y)*K2**n*(np.exp(2*n*x) - 4*np.exp(n*x)*K2**n + K2**(2*n))*n**3)/
      (np.exp(n*x) + K2**n)**4), (np.exp(n*x - y)*K2**n*(-np.exp(n*x) + K2**n)*n**2)/
     (np.exp(n*x) + K2**n)**3, 0], [(np.exp(n*x - y)*K2**n*(-np.exp(n*x) + K2**n)*n**2)/
     (np.exp(n*x) + K2**n)**3, -((np.exp(n*x - y)*K2**n*n)/(np.exp(n*x) + K2**n)**2), 0], 
   [0, 0, 0]], [[(np.exp(n*x - y)*K2**n*(-np.exp(n*x) + K2**n)*n**2)/(np.exp(n*x) + K2**n)**3, 
    -((np.exp(n*x - y)*K2**n*n)/(np.exp(n*x) + K2**n)**2), 0], 
   [-((np.exp(n*x - y)*K2**n*n)/(np.exp(n*x) + K2**n)**2), -(K2**n/(np.exp(y)*(np.exp(n*x) + K2**n))), 
    0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]], 
 [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], 
  [[0, 0, 0], [0, -((np.exp(n*y - z)*K3**n*(np.exp(2*n*y) - 4*np.exp(n*y)*K3**n + K3**(2*n))*
       n**3)/(np.exp(n*y) + K3**n)**4), (np.exp(n*y - z)*K3**n*(-np.exp(n*y) + K3**n)*n**2)/
     (np.exp(n*y) + K3**n)**3], [0, (np.exp(n*y - z)*K3**n*(-np.exp(n*y) + K3**n)*n**2)/
     (np.exp(n*y) + K3**n)**3, -((np.exp(n*y - z)*K3**n*n)/(np.exp(n*y) + K3**n)**2)]], 
  [[0, 0, 0], [0, (np.exp(n*y - z)*K3**n*(-np.exp(n*y) + K3**n)*n**2)/(np.exp(n*y) + K3**n)**3, 
    -((np.exp(n*y - z)*K3**n*n)/(np.exp(n*y) + K3**n)**2)], 
   [0, -((np.exp(n*y - z)*K3**n*n)/(np.exp(n*y) + K3**n)**2), 
    -(K3**n/(np.exp(z)*(np.exp(n*y) + K3**n)))]]]]
		return self.ts*np.array(res)
	
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
		ja = self.geneJacobian(x)
		m = self.geneMango(x)
		
		f = self.geneFunction(x) + 0.5*np.tensordot(h, c, 2)
		fp = ja + 0.5*np.tensordot(m, c, 2)
		g = np.dot(fp, c) + np.dot(c, fp.T) + self.N
		
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
	