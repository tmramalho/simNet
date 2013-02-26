'''
Created on Dec 30, 2011

@author: tiago
'''

import numpy as np
import scipy.optimize as sop
import math
import adolc

class stochSolver():
	'''
	Solves the equations for a given model
	and returns the resulting path
	'''


	def __init__(self, dim, sigma):
		'''
		Constructor
		'''
		self.dim = dim
		self.m = np.zeros(dim, dtype=float)
		self.N = np.diag(np.ones(dim, dtype=float)*sigma)
		self.sigma = sigma
		
	def setInit(self, yi, rand=False):
		'''Set the initial point for the simulation'''
		self.yi = yi
		self.rand = rand
		
	def getInit(self):
		if(self.rand):
			return self.yi+np.random.multivariate_normal(self.m,self.N)
		else:
			return self.yi
			
	def solveLinearLangevin(self, A, dt, steps):
		'''Solve the langevin equation for the linear model'''
		sdt = math.sqrt(dt)
		y = np.zeros((steps,self.dim))
		y[0] = self.getInit()
		for i in range(1, steps):
			xsi = np.random.multivariate_normal(self.m,self.N)
			y[i] = y[i-1] + dt*np.dot(A,y[i-1]) + sdt*xsi
		return y
			
	def solveLinearPred(self, A, dt, steps):
		'''Predict the average value for the linear model'''
		y = np.zeros((steps,self.dim))
		c = np.zeros((steps, self.dim, self.dim), dtype=float)
		y[0] = self.yi
		c[0] = self.N
		At = np.transpose(A)
		'''adams setup'''
		f = np.zeros((4, self.dim))
		g = np.zeros((4, self.dim, self.dim))
		for i in range(1, 4):
			term = np.eye(self.dim) + dt*(np.dot(c[i-1],np.dot(At,np.linalg.inv(c[i-1]))) + A)
			f[i-1] = np.dot(term,np.dot(A,y[i-1]))
			ys = y[i-1] + dt*f[i-1]
			fs = np.dot(term,np.dot(A,ys))
			y[i] = y[i-1] + dt*fs
			
			g[i-1] = np.dot(A, c[i-1]) + np.dot(c[i-1], At) + self.N
			cs = c[i-1] + dt*g[3]
			gs = np.dot(A, cs) + np.dot(cs, At) + self.N
			c[i] = c[i-1] + dt*gs
			
		for i in range(4, steps):
			
			term = np.eye(self.dim) + dt*(np.dot(c[i-1],np.dot(At,np.linalg.inv(c[i-1]))) + A)
			f[3] = np.dot(term,np.dot(A,y[i-1]))
			ys = y[i-1] + 55*dt/24*f[3] - 59*dt/24*f[2] + 37*dt/24*f[1] - 9*dt/24*f[0]
			fs = np.dot(term,np.dot(A,ys))
			y[i] = y[i-1] + 9*dt/24*fs+19*dt/24*f[3]-5*dt/24*f[2]+dt/24*f[1]
			
			g[3] = np.dot(A, c[i-1]) + np.dot(c[i-1], At) + self.N
			cs = c[i-1] + 55*dt/24*g[3] - 59*dt/24*g[2] + 37*dt/24*g[1] - 9*dt/24*g[0]
			gs = np.dot(A, cs) + np.dot(cs, At) + self.N
			c[i] = c[i-1] + 9*dt/24*gs+19*dt/24*g[3]-5*dt/24*g[2]+dt/24*g[1]
			
			f = np.roll(f, -1, axis=0)
			g = np.roll(g, -1, axis=0)
			
		return (y,c)
	
	def solveLogLangevin(self, A, dt, steps):
		'''Solve the langevin equation for the log model'''
		sdt = math.sqrt(dt)
		y = np.zeros((steps,self.dim))
		y[0] = np.log(self.getInit())
		for i in range(1, steps):
			xsi = np.random.multivariate_normal(self.m,self.N)
			y[i] = y[i-1] + np.exp(-y[i-1])*dt*np.dot(A,np.exp(y[i-1])) + sdt*xsi
		return np.exp(y)
	
	def solveLinearLangevinPoisson(self, A, dt, steps):
		'''Solve the langevin equation for the linear model
		with poisson noise'''
		sdt = math.sqrt(dt)
		y = np.zeros((steps,self.dim))
		y[0] = self.getInit()
		for i in range(1, steps):
			xsi = np.random.multivariate_normal(self.m,self.sigma*np.diag(y[i-1]))
			y[i] = y[i-1] + dt*np.dot(A,y[i-1]) + sdt*xsi
		return y
	
	def solvePoissonPred(self, A, dt, steps):
		'''Predict the average value for the linear model'''
		y = np.zeros((steps,self.dim))
		c = np.zeros((steps, self.dim, self.dim), dtype=float)
		y[0] = self.yi
		c[0] = self.N
		At = np.transpose(A)
		for i in range(1, steps):
			factor = np.dot(A, c[i-1]) + np.dot(c[i-1], At)
			c[i] = c[i-1] + dt*factor + dt*self.N*y[i-1]
			term = np.dot(c[i-1],np.dot(At,np.linalg.inv(c[i-1]))) + A
			f = np.dot(A,y[i-1])
			y[i] = y[i-1] + dt*f + dt*dt*np.dot(term, f)
		return (y,c)
	
	def setOscillatorParameters(self, A, w, mu, y0):
		self.A = A
		self.w = w
		self.mu = mu
		self.y0 = y0
	
	def solveOscillatorLangevin(self, dt, steps):
		'''Solve the langevin equation for the couple oscillator'''
		sdt = math.sqrt(dt)
		x = np.zeros((steps,self.dim))
		y = np.zeros((steps,self.dim))
		x[0] = self.getInit()
		y[0] = self.y0
		for i in range(1, steps):
			xsi = np.random.multivariate_normal(self.m,self.N)
			x[i] = x[i-1] + dt*y[i-1] + sdt*xsi
			xa = np.tile(x[i-1],len(x[i])).reshape((len(x[i]),len(x[i])))
			dx = xa - xa.transpose()
			y[i] = y[i-1] + dt*(self.mu*(1-x[i-1]*x[i-1])*y[i-1]-self.w*self.w*x[i-1]+
							np.dot(self.A,dx.T).diagonal()) + sdt*xsi
		return (x,y)
	
	def oscillatorFunction(self, x):
		l = len(x)/2
		res = np.zeros((len(x)), dtype=x.dtype)
		'''position calc \dot{x} = y '''
		res[l:] = x[:l]
		xa = np.tile(x[l:],l).reshape((l,l))
		dx = xa - xa.transpose()
		'''velocity calc \dot{y} = m*(1-x^2)*y-w^2*x+A.dx'''
		res[:l] = (self.mu*(1-x[l:]*x[l:])*x[:l]-self.w*self.w*x[l:] +
				np.dot(self.A,dx.T).diagonal())
		return res
	
	def oscillatorJacobian(self, x):
		l = len(x)/2
		res = np.zeros((len(x),len(x)))
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
		res = np.zeros((len(x),len(x),len(x)))
		for i in range(0,l):
			'''only two nonzero entries'''
			res[i][i+l][i+l] = -2*self.mu*x[i] 
			'''symmetry'''
			res[i][i][i+l] = res[i][i+l][i] = -2*self.mu*x[i+l]
		return res
	
	def oscillatorMango(self, x):
		l = len(x)/2
		res = np.zeros((len(x),len(x),len(x),len(x)))
		for i in range(0,l):
			'''symmetry'''
			res[i][i][i+l][i+l] = res[i][i+l][i][i+l] = res[i][i+l][i+l][i] = -2*self.mu
		return res
	
	def bdfOscRootEquation(self, wi, wi1, wi2, NM, dt):
		x = wi[:self.dim*2]
		c = wi[self.dim*2:].reshape((self.dim*2,self.dim*2))
		
		h = self.oscillatorHessian(x)
		ja = adolc.jacobian(1, x)
		m = self.oscillatorMango(x)
		
		fx = self.oscillatorFunction(x) + 0.5*np.tensordot(h, c, 2)
		fp = ja + 0.5*np.tensordot(m, c, 2)
		fc = np.dot(fp, c) + np.dot(c, fp.T) + NM
		
		fw = np.concatenate((fx, fc.flatten()))
		
		'''BDF formula'''
		return wi - 4*wi1/3 + wi2/3 - 2*dt*fw/3
		
	def solveOscillatorDet(self, dt, steps):
		'''Predict the average value for the coupled oscillator'''
		x = np.zeros((steps,self.dim*2))
		x[0,:self.dim] = self.y0 #velocities
		x[0,self.dim:] = self.getInit() #positions
		'''inverse covariance matrix'''
		c = np.zeros((steps, self.dim*2, self.dim*2), dtype=float)
		NM = np.diag(np.ones(self.dim*2, dtype=float)*self.sigma)
		c[0] = NM
		'''adolc: tape a function's path'''
		ax = np.array([adolc.adouble(0) for n in range(self.dim*2)])
		adolc.trace_on(1)
		adolc.independent(ax)
		ay = self.oscillatorFunction(ax)
		adolc.dependent(ay)
		adolc.trace_off()
		'''BDF setup'''
		for i in range(1, 3):
			f = self.oscillatorFunction(x[i-1])
			xs = x[i-1] + dt*f
			fs = self.oscillatorFunction(xs)
			x[i] = x[i-1] + dt*fs
			
			fp = adolc.jacobian(1, x[i])
			g = np.dot(fp, c[i-1]) + np.dot(c[i-1], fp.T) + NM
			cs = c[i-1] + dt*g
			gs = np.dot(fp, cs) + np.dot(cs, fp.T) + NM
			c[i] = c[i-1] + dt*gs
			
		for i in range(3, steps):
			wi1 = np.concatenate((x[i-1],c[i-1].flatten()))
			wi2 = np.concatenate((x[i-2],c[i-2].flatten()))
			
			wi = sop.fsolve(self.bdfOscRootEquation, wi1, args=(wi1, wi2, NM, dt))
			
			x[i] = wi[:self.dim*2]
			c[i] = wi[self.dim*2:].reshape((self.dim*2,self.dim*2))
			
		return (x,c)
	