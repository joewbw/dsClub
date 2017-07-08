class NeuralNetworkErrorPropagator:

    def backPropNetworkErrors(self, network, output):
        eta = network.eta
        alpha = network.alpha
        inputNeurns = neuralNetwork.inputLayer
        hiddenNeurons = neuralNetwork.hiddenLayer
        outputNeurons = neuralNetwork.outputLayer
        activationFunction = neuralNetwork.activationFunction
        prediction = outputNeurons.[0].activation

        # compute activations and errors
        # backprop output to hidden layer
        # update weights

    def computeOutputActivationsAndErrors(self, outputNeurons, expectedOutput):
        pass

    def backPropOutputToHiddenLayer(self, activationFunction, hiddenLayer, outputLayer):
        pass

    def updateWeights(self, inputLayer, outputLayer, eta, alpha):
        pass
