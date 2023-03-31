import numpy as np
from matplotlib import pyplot as plt
import sympy as sym


class OLSRegression():
    def __init__(self, x, y):
        self.y = y
        self.x = x
        self.betaTwo = self.betaTwoCalc()
        self.betaOne = self.betaOneCalc(self.betaTwo)
        self.plt = plt

    def dataPlot(self):
        self.plt.xlabel("birth")
        self.plt.ylabel("protein")
        self.plt.scatter(self.x, self.y)

    def betaTwoCalc(self):
        sumXiYi = np.sum((self.x - self.x.mean()) * (self.y - self.y.mean()))
        sumXiSquare = np.sum(self.x ** 2) - self.x.size * self.x.mean() ** 2

        print(111111, sumXiYi, sumXiSquare)
        return sumXiYi / sumXiSquare

    def betaOneCalc(self, beta2):
        return self.y.mean() - self.x.mean() * beta2

    def regressionLinePlot(self):
        x = sym.Symbol("x")
        y = self.betaOne + self.betaTwo * x
        range = np.arange(0, 65)
        f = sym.lambdify(x, y, "numpy")
        plt.plot(range, f(range), label="regression")

    def coefficientOfCorrelationCalc(self):
        covarince = np.sum((self.x - self.x.mean()) * (self.y - self.y.mean()))
        thetaX = (np.sum((self.x - self.x.mean()) ** 2)) ** 0.5
        thetaY = (np.sum((self.y - self.y.mean()) ** 2)) ** 0.5
        return covarince / thetaY / thetaX

    def prediction(self, xToPredict):
        x = sym.Symbol("x")
        y = self.betaOne + self.betaTwo * x
        return y.subs(x, xToPredict)

#
# q2x = np.array([4.7, 7.5, 8.7, 9.7, 11.2, 15.2, 15.2, 16.8, 37.3, 46.7, 59.1, 59.9, 61.4, 62.6])
# q2y = np.array([45.6, 39.7, 33, 27.0, 25.9, 23.5, 23.4, 22.2, 20, 19.1, 18.3, 18, 17.9, 15])
# q2 = OLSRegression(q2x, q2y)
# q2.dataPlot()
# q2.regressionLinePlot()
# print(f"Beta1 = {q2.betaOne} and Beta2 = {q2.betaTwo}")
# print(
#     f"The cofficient of correlation is {q2.coefficientOfCorrelationCalc()}, R^2 = {q2.coefficientOfCorrelationCalc() ** 2}")
# print(f"The prediction of I is {q2.prediction(37.3)}")
# print(f"The prediction of mean is {q2.prediction(q2.x.mean())}")

q3x = np.array([10.5, 6.0, 8.7, 9.3, 11.8, 7.5, 15, 6.3, 8.5, 5.4])
q3y = np.array([17.3, 14, 19.1, 14.5, 20, 16.3, 23.8, 14, 17.3, 13.3])


q3 = OLSRegression(q3x,q3y)
q3.dataPlot()
q3.regressionLinePlot()
print(f"Beta1 = {q3.betaOne} and Beta2 = {q3.betaTwo}")

print(f"The cofficient of correlation is {q3.coefficientOfCorrelationCalc()} and R^2 is {q3.coefficientOfCorrelationCalc()**2}")
print(f"The prediction of 1000 is {q3.prediction(10)}")
q3.plt.show()
