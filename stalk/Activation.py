# -*- coding: utf-8 -*-

import theano
import theano.tensor as T
import numpy as np

class Activation():
	def __init__(self,type):
		if type=='sigmoid':
			self.function=T.nnet.sigmoid
		elif type=='tanh':
			self.function=T.tanh


