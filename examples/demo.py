# -*- coding: utf-8 -*-

from stalk.Layer import *
from stalk.load_data import *
from stalk.Model import *


def main():

	# データを読み込み
	datasets = load_data('mnist.pkl.gz')
	train_set_x, train_set_y = datasets[0]
	valid_set_x, valid_set_y = datasets[1]

	# ニューラルネットワークを作成
	layers=[]
	layers.append(FullyConnect(n_in=28 * 28, n_out=100, activation='tanh'))
	layers.append(FullyConnect(n_in=100, n_out=50, activation='tanh'))
	layers.append(LogisticRegression(n_in=50, n_out=10))

	# モデルをコンパイル，学習
	model = Model(layers)
	model.fit(train_set_x, train_set_y,validation_data=[valid_set_x,valid_set_y])

if __name__=="__main__":
	main()