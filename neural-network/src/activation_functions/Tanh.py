import math

class Tanh:

    def compute(self,input):
        return (2 / (1 + math.exp(-2 * input))) - 1

    def computeInverse(self,input):
        return 1 - (input * input)
