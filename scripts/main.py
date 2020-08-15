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


df = pd.read_csv(("data/house_data.csv") , nrows=200)

#convert it to square meters
spaceDf = df["sqft_living"].apply( lambda x : x / 10.764).apply( lambda x : np.round(x , 1))

#normalize features/predictor
yearDf = normalizeDataframeColumn(df["yr_built"])
priceDf = normalizeDataframeColumn(df["price"])

a0 = 0.5
b0 = 1
alpha = 0.2
max_iterations = 2000

def point_error (x , y , a , b):
    return (a +b*x) - y

def cost (X , Y , a , b):

    cost = 0
    for i in range(len(X)):
        cost = point_error(X[i] , Y[i] , a , b)**2 
    return cost


def gradient_descent(X , Y , theta , alpha):
    a_deriv = 0
    b_deriv = 0

    N = len(X)
    normalization = alpha/float(N)
    for i in range(N):
        err = point_error(X[i] , Y[i] , theta[0] , theta[1] )
        a_deriv = err
        b_deriv = err*X[i]

    theta[0] = theta[0] - normalization*a_deriv
    theta[1] = theta[1] - normalization*b_deriv

    return theta

def onProgress(iterations , cost):
    print("[{}/{}] => cost={}".format(iterations+1 , max_iterations , cost))

def onEnd(iterations , theta):
        print("a={} , b={} with a0={} , b0={} , alpha={} in {} iterations".format(theta[0] , theta[1] , a0 , b0 , alpha , iterations))


@timed
def main (progressCallback , endCallback):
    theta = [a0 , b0]
    J = []
    iterations = 0
    last_J = 0
    while True:

        progressCallback(iterations , last_J)

        theta = gradient_descent(yearDf , priceDf , theta  , alpha)
        iterations += 1

        last_J = cost(yearDf , priceDf , theta[0] , theta[1])
        J.append(last_J)

        if np.isclose(last_J , 0 , atol=1e-04) or iterations == max_iterations:
            break


    endCallback(iterations , theta)

    return J


J = main(onProgress , onEnd)

plt.plot(J , "b.")
plt.show()

try:
    quit()
except:
    print("Bye!")