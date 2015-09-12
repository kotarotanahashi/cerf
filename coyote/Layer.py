# -*- coding: utf-8 -*-

from coyote.Activation import *
import numpy as np
import theano.tensor as T
import theano
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

class Layer(object):
	def __init__(self):
		self.n_in=None
		self.n_out=None
		self.input=T.matrix()
		self.output=None
		self.batch_size = None

	def setInput(self,pre_output):
		self.input = pre_output

	def init_params(self):
		pass

	def set_batchsize(self,batch_size):
		self.batch_size = batch_size


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

	def __init__(self, n_out):
		super(LogisticRegression,self).__init__()
		self.n_in = None
		self.n_out = n_out

	def init_params(self):
		# initialize with 0 the weights W as a matrix of shape (n_in, n_out)
		self.W = theano.shared(
			value=np.zeros(
				(self.n_in, self.n_out),
				dtype=theano.config.floatX
			),
			name='W',
			borrow=True
		)
		# initialize the biases b as a vector of n_out 0s
		self.b = theano.shared(
			value=np.zeros(
				(self.n_out,),
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

class ImageInput(Layer):
	def __init__(self, out_sx, out_sy, out_depth=1):
		super(ImageInput,self).__init__()
		#self.input = T.matrix()
		self.input = None
		self.out_depth = out_depth
		self.out_sx = out_sx
		self.out_sy = out_sy
		self.image_shape = None

	def set_batchsize(self,batch_size):
		self.batch_size = batch_size
		self.image_shape = (batch_size, self.out_depth, self.out_sx, self.out_sy)


	def get_output(self):
		return self.input

	def get_output_image_shape(self):
		return self.image_shape




class LeNetConvPoolLayer(Layer):
	"""Pool Layer of a convolutional network """

	def __init__(self, out_depth, filter_size, poolsize=(2, 2)):
		super(LeNetConvPoolLayer,self).__init__()
		self.image_shape = None
		self.in_depth = None
		self.input = T.matrix()
		self.rng = np.random.RandomState(4711)

		self.poolsize = poolsize
		self.filter_size = filter_size
		self.out_depth = out_depth

	def init_params(self):
		self.filter_shape = (self.out_depth, self.in_depth, self.filter_size, self.filter_size)
		# there are "num input feature maps * filter height * filter width"
		# inputs to each hidden unit
		fan_in = np.prod(self.filter_shape[1:])
		# each unit in the lower layer receives a gradient from:
		# "num output feature maps * filter height * filter width" /
		#   pooling size
		fan_out = (self.filter_shape[0] * np.prod(self.filter_shape[2:]) /
					np.prod(self.poolsize))
		# initialize weights with random weights
		W_bound = np.sqrt(6. / (fan_in + fan_out))
		self.W = theano.shared(
			np.asarray(
				self.rng.uniform(low=-W_bound, high=W_bound, size=self.filter_shape),
				dtype=theano.config.floatX
			),
			borrow=True
		)

		# the bias is a 1D tensor -- one bias per output feature map
		b_values = np.zeros((self.filter_shape[0],), dtype=theano.config.floatX)
		self.b = theano.shared(value=b_values, borrow=True)

		# store parameters of this layer
		self.params = [self.W, self.b]

	def set_input_image_shape(self, image_shape):
		self.image_shape = image_shape

	def get_output_image_shape(self):
		conv_out_shape = self.image_shape[2] - self.filter_size + 1, self.image_shape[3] - self.filter_size + 1
		return self.batch_size, self.out_depth, conv_out_shape[0]/self.poolsize[0], conv_out_shape[1]/self.poolsize[1]

	def get_output(self):
		# convolve input feature maps with filters
		print "self.image_shape",self.image_shape
		self.conv_out = conv.conv2d(
			input=self.input,
			filters=self.W,
			filter_shape=self.filter_shape,
			image_shape=self.image_shape
		)

		# downsample each feature map individually, using maxpooling
		self.pooled_out = downsample.max_pool_2d(
			input=self.conv_out,
			ds=self.poolsize,
			ignore_border=True
		)

		return T.tanh(self.pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

class Flattern(Layer):
	def __init__(self):
		super(Flattern,self).__init__()
		self.input = T.matrix()
		self.image_shape = None
		self.in_depth = None

	def set_input_image_shape(self, image_shape):
		self.image_shape = image_shape
		self.n_out = self.image_shape[2] * self.image_shape[3] * self.in_depth

	def get_output(self):
		return self.input.flatten(2)