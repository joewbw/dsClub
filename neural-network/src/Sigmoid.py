import math

class Sigmoid:

    def compute(self,input):
        return (1.0 / (1.0 + math.exp(-input)))

    def computeInverse(self,input):
        return input * (1.0 - input)
