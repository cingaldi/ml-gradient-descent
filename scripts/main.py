import pandas as pd
import numpy as np
import sys

import matplotlib.pyplot as plt

from perf import timed


def normalizeDataframeColumn(column):
    return (column - column.min()) / (column.max() - column.min())

def getLine(min , max , a , b):
    x = np.linspace(min , max , 10)
    y = a + b*x
    return a , b


df = pd.read_csv(("data/house_data.csv"))

#convert it to square meters
spaceDf = df["sqft_living"].apply( lambda x : x / 10.764).apply( lambda x : np.round(x , 1))

#normalize features/predictor
yearDf = normalizeDataframeColumn(df["yr_built"])
priceDf = normalizeDataframeColumn(df["price"])

a0 = 0.5
b0 = 1
alpha = 0.1
max_iterations = 2000

def point_error (X , y , theta):
    return sum(X * theta) - y

def cost (X , Y , theta):

    cost = 0
    for i in range(len(X)):
        cost = point_error(X[i] , Y[i] , theta)**2 
    return cost


def gradient_descent(X , Y , theta , alpha):

    examples , features = np.shape(X)
    deriv = np.zeros(features)

    normalization = alpha/float(examples)
    for i in range(examples):
        err = point_error(X[i] , Y[i] , theta )
        deriv += np.multiply(X[i] , err)

    theta = theta - np.multiply(deriv , normalization)

    return theta

J = []
def onProgress(iterations , cost):
    J.append(cost)
    print("[{}/{}] => cost={}".format(iterations , max_iterations , cost))

def onEnd(iterations , theta):
        print("a={} , b={} with a0={} , b0={} , alpha={} in {} iterations".format(theta[0] , theta[1] , a0 , b0 , alpha , iterations))


@timed
def main (progressCallback , endCallback):
    theta = [a0 , b0]
    last_J = 0
    iterations = 0
    X = np.transpose([np.ones(len(yearDf) , dtype=None) , yearDf])
    while True:


        theta = gradient_descent(X , priceDf , theta  , alpha)
        iterations += 1

        new_J = cost(X , priceDf , theta)

        if np.isclose(abs(last_J  - new_J), 0 , atol=1e-03) or iterations == max_iterations:
            break

        progressCallback(iterations , new_J)

        last_J = new_J


    endCallback(iterations , theta)


main(onProgress , onEnd)

plt.plot(J , "b.")
plt.show()

try:
    quit()
except:
    print("Bye!")