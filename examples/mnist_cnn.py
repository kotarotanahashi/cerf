# -*- coding: utf-8 -*-
import sys
sys.path.append('../../coyote')
from coyote.Layer import *
from coyote.load_data import *
from coyote.Model import *


def main():

	# load data
	datasets = load_data('mnist.pkl.gz')
	train_set_x, train_set_y = datasets[0]
	valid_set_x, valid_set_y = datasets[1]

	# make neural network
	layers = []
	layers.append(ImageInput(out_sx=28, out_sy=28, out_depth=1))
	layers.append(LeNetConvPoolLayer(out_depth=7, filter_size=5))
	layers.append(LeNetConvPoolLayer(out_depth=3, filter_size=5))
	layers.append(Flattern())
	layers.append(LogisticRegression(n_out=10))

	# compile model and train
	model = Model(layers)
	model.fit(train_set_x, train_set_y,validation_data=[valid_set_x,valid_set_y])

if __name__=="__main__":
	main()