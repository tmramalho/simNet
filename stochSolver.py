'''
Created on Dec 30, 2011

@author: tiago
'''

from odeSolver import odeSolver
import numpy as np
import math
import pycppad as pcad

class stochSolver():
	'''
	Solves the equations for a given model
	and returns the resulting path
	'''


	def __init__(self, dim, sigma, rand=False):
		'''
		Constructor
		'''
		self.dim = dim
		self.m = np.zeros(dim, dtype=float)
		self.N = np.eye(dim)*sigma
		self.sigma = sigma
		if rand:
			self.rand = 1
		else:
			self.rand = 0
			
	def setLinearParameters(self, A, y0):
		self.A = A
		self.y0 = y0
			
	def solveLinearLangevin(self, time, dt, steps):
		'''Solve the langevin equation for the linear model'''
		sdt = math.sqrt(dt)
		y = np.zeros((steps,self.dim))
		y[0] = self.y0 + self.rand*np.random.multivariate_normal(self.m,self.N)
		yn = np.copy(y[0])
		yp = np.copy(y[0])
		jt = int(time/float(steps*dt))
		for i in range(1, steps):
			for _ in range(0, jt):
				yp = np.copy(yn)
				xsi = np.random.multivariate_normal(self.m,self.N)
				yn = yp + dt*np.dot(self.A, yp) + sdt*xsi
			y[i] = np.copy(yn)
		return y
	
	def linearFunction(self, x, c):
		fs = np.dot(self.A, x)
		gs = np.dot(self.A, c) + np.dot(c, self.A) + self.N
		
		return(fs, gs)
		
	def solveLinearPred(self, time, dt, steps):
		'''Predict the average value for the linear model'''
		x = self.y0
		if self.rand:
			c = self.N
		else:
			c = np.zeros(self.dim)
		
		od = odeSolver()
		
		return od.adamsSolver(time, dt, steps, self.dim, float, 
							self.linearFunction, x, c)
	
	def solveLogLangevin(self, time, dt, steps):
		'''Solve the langevin equation for the log model'''
		sdt = math.sqrt(dt)
		y = np.zeros((steps,self.dim))
		y[0] = np.log(self.y0 + self.rand*np.random.multivariate_normal(self.m,self.N))
		for i in range(1, steps):
			xsi = np.random.multivariate_normal(self.m,self.N)
			y[i] = y[i-1] + np.exp(-y[i-1])*dt*np.dot(self.A,np.exp(y[i-1])) + sdt*xsi
		return np.exp(y)
	
	def solveLinearLangevinPoisson(self, time, dt, steps):
		'''Solve the langevin equation for the linear model
		with poisson noise'''
		sdt = math.sqrt(dt)
		y = np.zeros((steps,self.dim))
		y[0] = self.y0 + self.rand*np.random.multivariate_normal(self.m,self.N)
		for i in range(1, steps):
			xsi = np.random.multivariate_normal(self.m,self.sigma*np.diag(y[i-1]))
			y[i] = y[i-1] + dt*np.dot(self.A,y[i-1]) + sdt*xsi
		return y
	
	def solvePoissonPred(self, time, dt, steps):
		'''Predict the average value for the linear model'''
		y = np.zeros((steps,self.dim))
		c = np.zeros((steps, self.dim, self.dim), dtype=float)
		y[0] = self.y0
		c[0] = self.N
		At = np.transpose(self.A)
		for i in range(1, steps):
			factor = np.dot(self.A, c[i-1]) + np.dot(c[i-1], At)
			c[i] = c[i-1] + dt*factor + dt*self.N*y[i-1]
			term = np.dot(c[i-1],np.dot(At,np.linalg.inv(c[i-1]))) + self.A
			f = np.dot(self.A,y[i-1])
			y[i] = y[i-1] + dt*f + dt*dt*np.dot(term, f)
		return (y,c)
	
	def setOscillatorParameters(self, A, w, mu, y0, x0):
		self.A = A
		self.w = w
		self.mu = mu
		self.y0 = y0
		self.x0 = x0
		self.dtype = x0.dtype
		
	def setPackedOscillatorParameters(self, par, dtype):
		self.dtype = dtype #dataType for AD
		pos = 0
		dd=self.dim*self.dim
		self.A = par[pos:dd].reshape((self.dim,self.dim))
		pos = dd
		self.w = par[pos:pos+self.dim]
		pos += self.dim
		self.mu = par[pos:pos+1]
		pos += 1
		self.y0 = par[pos:pos+self.dim]
		pos += self.dim
		self.x0 = par[pos:pos+self.dim]
	
	def solveOscillatorLangevin(self, time, dt, steps):
		'''Solve the langevin equation for the couple oscillator'''
		sdt = math.sqrt(dt)
		x = np.zeros((steps,self.dim))
		y = np.zeros((steps,self.dim))
		x[0] = self.x0 + np.random.multivariate_normal(self.m,self.N)
		y[0] = self.y0 + np.random.multivariate_normal(self.m,self.N)
		yn = np.copy(y[0])
		yp = np.copy(y[0])
		xn = np.copy(x[0])
		xp = np.copy(x[0])
		jt = int(time/float(steps*dt))
		for i in range(1, steps):
			for _ in range(0, jt):
				xp = np.copy(xn)
				yp = np.copy(yn)
				xsi = np.random.multivariate_normal(self.m,self.N)
				xn = xp + dt*yp + sdt*xsi
				xsi = np.random.multivariate_normal(self.m,self.N)
				xa = np.tile(xp,(len(xp),1))
				dx = xa - xa.transpose()
				yn = yp + dt*(self.mu*(1-xp*xp)*yp-self.w*self.w*xp+
							np.dot(self.A,dx.T).diagonal()) + sdt*xsi
			y[i] = np.copy(yn)
			x[i] = np.copy(xn)
		return x
	
	def oscillatorFunction(self, x):
		lx = x.size
		l = lx/2
		res = pcad.ad(np.zeros(np.shape(x)))
		'''position calc \dot{x} = y '''
		res[l:] = x[:l]
		xa = np.tile(x[l:],(l,1))
		dx = xa - xa.transpose()
		'''velocity calc \dot{y} = m*(1-x^2)*y-w^2*x+A.dx'''
		try:
			res[:l] = (self.mu*(1-x[l:]*x[l:])*x[:l]-self.w*self.w*x[l:] +
				np.dot(self.A,dx.T).diagonal())
		except RuntimeWarning:
			print self.A, self.mu, self.w
		return res
	
	def oscillatorJacobian(self, x):
		l = len(x)/2
		res = np.zeros((len(x),len(x)), dtype=self.dtype)
		for i in range(0,l):
			res[i+l][i] = 1
			for j in range(0,2*l):
				if j == i:
					res[i][j] = self.mu*(1-x[i+l]*x[i+l])
				if j-l == i:
					res[i][j] = -2*self.mu*x[i]*x[j]-self.w[i]*self.w[i] 
					- np.sum(self.A,axis=1)[i] + self.A[i,i]
				if j >= l and j-l != i:
					res[i][j] = self.A[i,j-l]
		return res
	
	def oscillatorHessian(self, x):
		l = len(x)/2
		res = np.zeros((len(x),len(x),len(x)), dtype=self.dtype)
		for i in range(0,l):
			'''only two nonzero entries'''
			res[i][i+l][i+l] = -2*self.mu*x[i] 
			'''symmetry'''
			res[i][i][i+l] = res[i][i+l][i] = -2*self.mu*x[i+l]
		return res
	
	def oscillatorMango(self, x):
		l = len(x)/2
		res = np.zeros((len(x),len(x),len(x),len(x)), dtype=self.dtype)
		for i in range(0,l):
			'''symmetry'''
			res[i][i][i+l][i+l] = res[i][i+l][i][i+l] = res[i][i+l][i+l][i] = -2*self.mu
		return res
		
	def odeFunction(self, x, c):
		f = self.gf.forward(0, x)
		#ja = self.oscillatorJacobian(x)
		ja = self.gf.jacobian(x)
		h = self.oscillatorHessian(x)
		m = self.oscillatorMango(x)
		
		fs = f + 0.5*np.tensordot(h, c, 2)
		fp = ja + 0.5*np.tensordot(m, c, 2)
		gs = np.dot(fp, c) + np.dot(c, fp.T) + self.N
		
		return(fs, gs)
	
	def odeLinearFunction(self, x, c):
		fs = self.gf.forward(0, x)
		#fp = self.oscillatorJacobian(x)
		fp = self.gf.jacobian(x)
		gs = np.dot(fp, c) + np.dot(c, fp.T) + self.N
		
		return(fs, gs)
		
	def solveOscillatorDet(self, time, dt, steps):
		'''Predict the average value for the coupled oscillator'''
		x = np.zeros((self.dim*2), dtype=self.dtype)
		x[:self.dim] = self.y0 #velocities
		x[self.dim:] = self.x0 #positions
		'''inverse covariance matrix'''
		self.N = np.diag(np.ones(self.dim*2, dtype=self.dtype)*self.sigma)
		c = self.N
		
		'''cppad setup'''
		ax = pcad.independent(np.zeros(self.dim*2))
		ay = self.oscillatorFunction(ax)
		self.gf = pcad.adfun(ax, ay)
		self.gf.optimize()
		
		od = odeSolver()
		
		return od.adamsSolver(time, dt, steps, self.dim*2, self.dtype, 
							self.odeFunction, x, c)
		
	def solveOscillatorDeterministic(self, time, dt, steps):
		'''Predict the average value for the coupled oscillator'''
		x = np.zeros((self.dim*2), dtype=self.dtype)
		x[:self.dim] = self.y0 #velocities
		x[self.dim:] = self.x0 #positions
		'''inverse covariance matrix'''
		self.N = np.diag(np.ones(self.dim*2, dtype=self.dtype)*self.sigma)
		c = self.N
		
		od = odeSolver()
		
		return od.adamsSolver(time, dt, steps, self.dim*2, self.dtype, 
							self.odeLinearFunction, x, c)