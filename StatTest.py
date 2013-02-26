'''
Created on Jan 2, 2012

@author: tiago
'''

import warnings
import numpy as np
from stochSolver import stochSolver
from geneFixedSolver import geneFixedSolver
from extPlotter import extPlot

class test(object):
	'''
	Run statistical tests on the models
	'''


	def __init__(self, numRuns, time, dt, sigma, dim = 3, etype= 'linear', wd=None):
		'''
		Constructor
		'''
		self.numRuns = numRuns
		self.time = time
		self.dt = dt
		self.numSamples = 1000
		self.ds = time/float(self.numSamples)
		self.dim = dim
		self.ss = stochSolver(self.dim, sigma, False)
		self.gs = geneFixedSolver(self.dim, sigma)
		self.h = extPlot(wd)
		self.run = extPlot(wd)
		self.rawData = extPlot(wd)
		self.hm = extPlot(wd)
		self.pplot = extPlot(wd)
		self.pchi = extPlot(wd)
		self.workDir = wd
		self.solvers = {'linear' : self.ss.solveLinearLangevin,
						'log' : self.ss.solveLogLangevin,
						'poisson': self.ss.solveLinearLangevinPoisson,
						'osc': self.ss.solveOscillatorLangevin,
						'gene': self.gs.solveGeneticLangevin
						 }
		self.predictors = {'linear' : self.ss.solveLinearPred,
							'log' : self.ss.solveLinearPred,
							'poisson' : self.ss.solvePoissonPred,
							'osc' : self.ss.solveOscillatorDet,
							'gene' : self.gs.solveGeneticPred }
		self.modelType = etype
		self.so = self.solvers[etype]
		self.pred = self.predictors[etype]
		self.steps = int(self.time / self.ds)
		
	def simpleSimulation(self, plotRuns=False):
		'''Calculate some stochastic paths and calculate the moments'''
		xa = np.zeros((self.steps,self.dim))
		xs = np.zeros((self.steps,self.dim,self.dim))
		fp = np.zeros((self.numRuns, self.dim))
		
		for j in range(0,self.numRuns):
			x = self.so(self.time, self.dt, self.steps)
			xa += x
			xt = np.tile(x, (1, len(x[0]))).reshape(self.steps, self.dim, self.dim)
			xs += xt*xt.transpose((0,2,1))
			fp[j] = x[self.steps-1]
			if(plotRuns and j < 30):
				self.rawData.plotPath(x, self.time, self.ds)
				self.pplot.plotPaperRawTrajectories(x, self.time, self.ds)
			
		xa /= self.numRuns
		xat = np.tile(xa, (1, len(x[0]))).reshape(self.steps, self.dim, self.dim)
		xs = xs/self.numRuns - xat*xat.transpose((0,2,1))
		std = np.sqrt(np.diagonal(xs, axis1=1, axis2=2))
		self.simAverage = xa
		self.simCovariance = xs
		self.simStdDev = std
		self.simEndpoints = fp
		
		self.rawData.plotPath(xa, self.time, self.ds, a=1, lw=2, lstyle='-', cl='0.3')
		self.rawData.save('allRunsSimulation.pdf')
		
	def simplePrediction(self):
		'''Calculate predicted trajectories for linear model'''
		
		(x,c) = self.pred(self.time, self.dt, self.steps)

		self.predAverage = x[:,-self.dim:]
		self.predCovariance = c[:,-self.dim:,-self.dim:]
		self.predStdDev = np.sqrt(np.diagonal(c[:,-self.dim:,-self.dim:], axis1=1, axis2=2))
		if len(x[:,0]) > self.dim:
			self.velocities = True
			self.predAverageVel = x[:,:self.dim]
			self.predCovarianceVel = c[:,:self.dim,:self.dim]
			self.predStdDevVel = np.sqrt(np.diagonal(c[:,:self.dim,:self.dim], axis1=1, axis2=2))
				
		if self.modelType == 'osc':
			(xd, cd) = self.ss.solveOscillatorDeterministic(self.time, self.dt, self.steps)
			dev = np.sqrt(np.diagonal(cd[:,-self.dim:,-self.dim:], axis1=1, axis2=2))
			self.rawData.plotTrajectories(xd[:,-self.dim:], dev, self.time, self.ds, lw=1, cl='0.5', lim=[-4, 4])
			self.pplot.plotPaperTrajectories(self.simAverage, self.simStdDev, self.predAverage, self.predStdDev, xd[:,-self.dim:], dev, self.time, self.ds)
		elif self.modelType == 'gene':
			(xd, cd) = self.gs.solveGeneticDet(self.time, self.dt, self.steps)
			dev = np.sqrt(np.diagonal(cd[:,-self.dim:,-self.dim:], axis1=1, axis2=2))
			self.rawData.plotTrajectories(xd[:,-self.dim:], dev, self.time, self.ds, lw=1, cl='0.5')
			self.pplot.plotPaperTrajectories(self.simAverage, self.simStdDev, self.predAverage, self.predStdDev, xd[:,-self.dim:], dev, self.time, self.ds)
		self.rawData.plotTrajectories(self.predAverage, self.predStdDev, self.time, self.ds, lw= 1, cl='k')
		self.rawData.save('allRunsPrediction.pdf')
			
	def plotThings(self):
		self.run.plotTrajectories(self.simAverage, self.simStdDev, self.time, self.ds)
		self.run.save("simulatedTrajectory.pdf")
		self.run.clear()
		
		self.hm.plotHeatMap(self.simCovariance[self.steps-1])
		self.hm.save('simulatedCovariance.pdf')
		self.hm.clear()
		
		self.run.plotTrajectories(self.predAverage, self.predStdDev, self.time, self.ds)
		self.run.save("predictedTrajectory.pdf")
		self.run.clear()
		
		if self.velocities:
			self.run.plotTrajectories(self.predAverageVel, self.predStdDevVel, self.time, self.ds)
			self.run.save("predictedVelocities.pdf")
			self.run.clear()
		
		self.h.plotHistogramPoints(self.simEndpoints)
		self.h.plotHistogramPrediction(self.predAverage[self.steps-1], self.predStdDev[self.steps-1])
		self.h.save("finalHistogram.pdf")
		
		self.pchi.plotPaperHistogramPoints(self.simEndpoints)
		self.pchi.plotPaperHistogramPrediction(self.predAverage[self.steps-1], self.predStdDev[self.steps-1])
		self.pchi.save("paperHistogram.pdf")
		
		self.hm.plotHeatMap(self.predCovariance[self.steps-1])
		self.hm.save('predictedCovariance.pdf')
		
		self.pplot.save('paperTraj.pdf')
			
	def compareChiSquare(self):
		res = np.zeros(self.numRuns)
		ca = self.predAverage[-1]
		C = np.linalg.inv(self.predCovariance[-1])
		for i in range(0, self.numRuns):
			res[i] = np.dot(self.simEndpoints[i] - ca, np.dot(C, self.simEndpoints[i] - ca))
		chi = extPlot(self.workDir)
		chi.plotSingleHistogramPoints(res, self.dim)
		chi.save('chiTest.pdf')
		self.pchi.plotPaperSingleHistogramPoints(res, self.dim)
		self.pchi.save("paperHistogram.pdf")
		
	def klEvolution(self):
		res = np.zeros(self.numSamples)
		for i in range(1, self.numSamples):
			S = np.linalg.inv(self.simCovariance[i])
			d = self.simAverage[i] - self.predAverage[i]
			res[i] = 0.5*(np.trace(np.dot(S, self.predCovariance[i]))
					+ np.dot(d.T, np.dot(S, d)) 
					- np.log(np.linalg.det(self.predCovariance[i])/np.linalg.det(self.simCovariance[i]))
					- self.dim)
		kl = extPlot(self.workDir)
		kl.plotPath(res, self.time, self.ds)
		kl.save("klDivergence.pdf")

