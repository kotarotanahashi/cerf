# -*- coding: utf-8 -*-

from cerf.Activation import *
import numpy as np
import theano.tensor as T
import theano

class Layer(object):
	def __init__(self):
		self.n_in=None
		self.n_out=None
		self.input=T.matrix()
		self.output=None

	def setInput(self,pre_output):
		self.input = pre_output


class FullyConnect(Layer):
	def __init__(self,n_in=None,n_out=None,activation='tanh'):
		super(FullyConnect,self).__init__()

		self.n_in = n_in
		self.n_out = n_out
		self.activation = Activation(activation).function
		self.input=T.matrix()
		self.output=T.matrix()
		self.rng = np.random.RandomState(4711)

		W_values = np.asarray(
			self.rng.uniform(
				low=-np.sqrt(6. / (self.n_in + self.n_out)),
				high=np.sqrt(6. / (self.n_in + self.n_out)),
				size=(self.n_in, self.n_out)
			),
			dtype=theano.config.floatX
		)
		if self.activation == T.nnet.sigmoid:
			W_values *= 4
		self.W = theano.shared(value=W_values, name='W', borrow=True)

		b_values = np.zeros((self.n_out,), dtype=theano.config.floatX)
		self.b = theano.shared(value=b_values, name='b', borrow=True)

		# parameters of the model
		self.params = [self.W, self.b]

	def get_output(self):
		lin_output = T.dot(self.input, self.W) + self.b
		self.output = (
			lin_output if self.activation is None
			else self.activation(lin_output)
		)
		return self.output




class LogisticRegression(Layer):

	def __init__(self, n_in, n_out):
		super(LogisticRegression,self).__init__()

		# initialize with 0 the weights W as a matrix of shape (n_in, n_out)
		self.W = theano.shared(
			value=np.zeros(
				(n_in, n_out),
				dtype=theano.config.floatX
			),
			name='W',
			borrow=True
		)
		# initialize the biases b as a vector of n_out 0s
		self.b = theano.shared(
			value=np.zeros(
				(n_out,),
				dtype=theano.config.floatX
			),
			name='b',
			borrow=True
		)

		# symbolic description of how to compute prediction as class whose
		# probability is maximal
		# self.y_pred = T.argmax(self.p_y_given_x, axis=1)
		# end-snippet-1

		# parameters of the model
		self.params = [self.W, self.b]

		# keep track of model input
		self.input = None


	def get_cost(self, y):
		self.p_y_given_x = T.nnet.softmax(T.dot(self.input, self.W) + self.b)
		self.y_pred = T.argmax(self.p_y_given_x, axis=1)
		return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

	def errors(self, y):
		# check if y has same dimension of y_pred
		if y.ndim != self.y_pred.ndim:
			raise TypeError(
				'y should have the same shape as self.y_pred',
				('y', y.type, 'y_pred', self.y_pred.type)
			)

		# check if y is of the correct datatype
		if y.dtype.startswith('int'):
			# the T.neq operator returns a vector of 0s and 1s, where 1
			# represents a mistake in prediction
			return T.mean(T.neq(self.y_pred, y))
		else:
			raise NotImplementedError()