import random

class Neuron:

    def __init__(self):
        self.delta = random.uniform(0.0,1.0)
        self.activation = random.uniform(0.0,1.0)
        self.weight = random.uniform(0.0,1.0)
        self.deltaWeight = random.uniform(0.0,1.0)
        self.inputWeights = []
        self.inputDeltaWeights = []
