# -*- coding: utf-8 -*-


import theano.tensor as T
from theano import *
import numpy as np
from theano import shared

def main():
	x = T.dvector('x')

	print "xname:%s" % x.name
	y = T.dvector('y')

	a = shared(np.array([2,3]))

	z = T.dot(x,y)

	x2=x

	f=function([x2,y],z)
	print f(np.array([2,2]),np.array([3,4]))

	inc = T.iscalar('inc')
	state = shared(0)
	accumulator = function([inc], state, updates=[(state, state+inc)])
	

if __name__=="__main__":
	main()