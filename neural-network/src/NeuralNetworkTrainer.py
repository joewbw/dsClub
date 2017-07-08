import random

class NeuralNetworkTrainer:

    def __init__(self, neuralNetwork, dataset):
        self.dataset = dataset
        self.neuralNetwork = neuralNetwork

    def train(self):
        random.shuffle(self.dataset)
        for row in self.dataset:
            self.neuralNetwork.setInputLayerActivation(row)

            self.propagateBetweenLayers(self.neuralNetwork.inputLayer,
                                        self.neuralNetwork.hiddenLayer)

            self.propagateBetweenLayers(self.neuralNetwork.hiddenLayer,
                                        self.neuralNetwork.outputLayer)

            expected = row[2]
            actual = self.neuralNetwork.outputLayer[0].activation
            self.computeOutputActivations(self.neuralNetwork.outputLayer, [expected])

            self.backpropOutputToHiddenLayer(self.neuralNetwork.hiddenLayer,
                                             self.neuralNetwork.outputLayer)

            self.updateWeights(self.neuralNetwork.inputLayer,
                               self.neuralNetwork.hiddenLayer)

            self.updateWeights(self.neuralNetwork.hiddenLayer,
                               self.neuralNetwork.outputLayer)

            print "expected: " + str(expected) + ", actual: " + str(actual)

    def propagateBetweenLayers(self, inputNeurons, outputNeurons):
        for outputNeuron in outputNeurons:
            sum = outputNeuron.weight
            inputIndex = 0
            for inputNeuron in inputNeurons:
                sum += inputNeuron.activation * outputNeuron.inputWeights[inputIndex]
                inputIndex += 1
            outputNeuron.activation = self.neuralNetwork.activationFunction.compute(sum)

    def computeOutputActivations(self, outputNeurons, expectedOutput):
        index = 0
        for outputNeuron in outputNeurons:
            expected = expectedOutput[index]
            actual = outputNeuron.activation
            outputNeuron.delta = (expected - actual) * actual * (1.0 - actual)
            index += 1

    def backpropOutputToHiddenLayer(self, hiddenNeurons, outputNeurons):
        hiddenIndex = 0
        for hiddenNeuron in hiddenNeurons:
            hiddenSum = 0.0
            for outputNeuron in outputNeurons:
                hiddenSum += outputNeuron.inputWeights[hiddenIndex] * outputNeuron.delta
            hiddenNeuron.delta = hiddenSum * \
                self.neuralNetwork \
                    .activationFunction.computeInverse(hiddenNeuron.activation)
            hiddenIndex += 1

    def updateWeights(self, inputNeurons, outputNeurons):
        eta = self.neuralNetwork.eta
        alpha = self.neuralNetwork.alpha

        for outputNeuron in outputNeurons:
            outputNeuron.deltaWeight = \
                (eta * outputNeuron.delta) + (alpha * outputNeuron.deltaWeight)
            outputNeuron.weight = outputNeuron.weight + outputNeuron.deltaWeight

            index = 0
            for inputNeuron in inputNeurons:
                outputNeuron.inputDeltaWeights[index] = \
                    (eta * inputNeuron.activation * outputNeuron.delta) \
                    + (alpha * outputNeuron.inputDeltaWeights[index])
                outputNeuron.inputWeights[index] = \
                    outputNeuron.inputWeights[index] + outputNeuron.inputDeltaWeights[index]
                index += 1
