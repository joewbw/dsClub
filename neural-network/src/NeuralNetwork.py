import random

import Neuron as Neuron
import Sigmoid as Sigmoid

class NeuralNetwork:

    inputLayer  = []
    hiddenLayer = []
    outputLayer = []

    inputNeurons = 2
    hiddenNeurons = 3
    outputNeurons = 1

    alpha = 0.5
    eta = 0.8

    activationFunction = Sigmoid.Sigmoid()

    def __init__(self):
        self.createNeurons(self.inputLayer, self.inputNeurons, 0)
        self.createNeurons(self.hiddenLayer, self.hiddenNeurons, self.inputNeurons)
        self.createNeurons(self.outputLayer, self.outputNeurons, self.hiddenNeurons)

    def createNeurons(self, layer, layerNeurons, inputLayerNeurons):
        for x in xrange(layerNeurons):
            neuron = Neuron.Neuron()
            neuron.inputWeights = self.getRandomWeights(inputLayerNeurons)
            neuron.inputDeltaWeights = self.getRandomWeights(inputLayerNeurons)
            layer.append(neuron)

    def getRandomWeights(self, n):
        weights = []
        for i in range(n):
            weights.append(random.uniform(0,1))
        return weights

    def setInputLayerActivation(self, row):
        i = 0
        for neuron in self.inputLayer:
            neuron.activation = row[i]
            i+=1
