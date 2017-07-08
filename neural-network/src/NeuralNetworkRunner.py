#!/usr/bin/env python

import NeuralNetwork as NeuralNetwork
import NeuralNetworkTrainer as NeuralNetworkTrainer

def train(network, trainer, dataset):
    print "[Training]"
    trainer.train()

def classify(network, dataset):
    print "[Classifying]"

if __name__ == '__main__':

    print "================"
    print " Neural network "
    print "================"

    dataset = [
        [0.0,0.0,0.0],
        [0.0,1.0,1.0],
        [1.0,0.0,1.0],
        [1.0,1.0,0.0]
    ]

    network = NeuralNetwork.NeuralNetwork()
    trainer = NeuralNetworkTrainer.NeuralNetworkTrainer(network, dataset)

    for i in range(10000):
        print " ============================================================ "
        print " = Training EPOCH (" + str(i) + ") "
        print " ============================================================ "
        train(network, trainer, dataset)

    #classify(network, dataset)
