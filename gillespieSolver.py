'''
Created on Feb 16, 2012

@author: tiago
'''

import numpy as np

class gill(object):
	'''
	the Gillespie algorithm
	'''


	def __init__(self, time):
		'''
		Constructor
		'''
		
		self.time = time
		
	def setInit(self, yi, rand=False):
		'''Set the initial point for the simulation'''
		self.yi = yi
		self.rand = rand
		
	def getInit(self):
		if(self.rand):
			return self.yi+np.random.multivariate_normal(self.m,self.N)
		else:
			return self.yi
		
	def drawNextLinearReaction(self, y, H, an, a0, r):
		acc = 0
		for i in range(0, len(y)):
			for j in range(0,len(y)):
				acc += an[i,j]
				if(r < acc/a0):
					y[i] += H[i,j]
					return 1
		
	def solveLinearGillespie(self, A):
		values = [[],[]]
		t = 0
		y = self.getInit()
		values[0].append(t)
		values[1].append(y)
		C = np.abs(A)
		H = np.sign(A)
		while t < self.time:
			rn = np.random.uniform(size=2)
			'''rates are A_ik y_i y_k or A_ii y_i'''
			y1 = np.tile(y,len(y)).reshape((len(y),len(y)))
			an = C*y1
			a0 = np.sum(an)
			'''draw next time for reaction'''
			t += -np.log(rn[0])/a0
			'''draw next reaction'''
			self.drawNextLinearReaction(y, H, an, a0, rn[1])
			values[0].append(t)
			values[1].append(np.array(y))
		
		return values
			