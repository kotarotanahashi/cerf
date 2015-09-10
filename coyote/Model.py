# -*- coding: utf-8 -*-

import theano
import theano.tensor as T


class Model():

	def __init__(self,layers):

		self.x = T.matrix('x')
		self.y = T.ivector('y')
		self.learning_rate=0.05
		self.layers = layers

		self.layers[0].setInput(self.x)
		print "name:%s"%self.layers[0].input.name
		for i,layer in enumerate(self.layers[1:]):
			layer.setInput(self.layers[i].get_output())
		for layer in self.layers:
			try:
				self.params += layer.params
			except:
				self.params = layer.params
		self.cost = self.layers[-1].get_cost(self.y)
		self.grads = T.grad(self.cost, self.params)
		self.updates = [
			(param_i, param_i - self.learning_rate * grad_i)
			for param_i, grad_i in zip(self.params, self.grads)
		]



	def fit(self,train_set_x,train_set_y,batch_size=600,n_epochs=100,validation_data=None):

		index = T.lscalar()
		self.train_model = theano.function(
			[index],
			self.cost,
			updates=self.updates,
			givens={
				self.x: train_set_x[index * batch_size: (index + 1) * batch_size],
				self.y: train_set_y[index * batch_size: (index + 1) * batch_size]
			}
		)

		if validation_data:
			valid_set_x, valid_set_y = validation_data
			self.validate_model = theano.function(
				[index],
				self.layers[-1].errors(self.y),
				givens={
					self.x: valid_set_x[index * batch_size: (index + 1) * batch_size],
					self.y: valid_set_y[index * batch_size: (index + 1) * batch_size]
				}
			)



		n_train_batches = train_set_x.get_value(borrow=True).shape[0]/ batch_size
		n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]/ batch_size
		#n_train_batches=1
		epoch = 0
		done_looping = False
		while (epoch < n_epochs) and (not done_looping):
			epoch += 1
			for minibatch_index in xrange(n_train_batches):
				iter = (epoch - 1) * n_train_batches + minibatch_index

				cost_ij = self.train_model(minibatch_index)


				if iter % 1000 == 0:
					print 'training @ iter = ', iter
					print "cost:%f" % cost_ij
					if self.validate_model:
						print "accuracy:%f" % (1.0-self.validate_model(min(minibatch_index,n_valid_batches)))


