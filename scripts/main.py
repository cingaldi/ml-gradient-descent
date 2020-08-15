import pandas as pd
import numpy as np
import sys

import matplotlib.pyplot as plt

from perf import timed


def normalizeDataframeColumn(column):
    return (column - column.min()) / (column.max() - column.min())

def getLine(min , max , a , b):
    x = np.linspace(min , max , 100)
    y = a + b*x
    return a , b


df = pd.read_csv(("data/house_data.csv") , nrows=50)

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


def gradient_descent(X , Y , a , b , alpha):
    a_deriv = 0
    b_deriv = 0

    N = len(X)
    normalization = alpha/float(N)
    for i in range(N):
        a_deriv = point_error(X[i] , Y[i] , a , b )
        b_deriv = point_error(X[i] , Y[i] , a , b )*X[i]

    a = a - normalization*a_deriv
    b = b - normalization*b_deriv
    return a , b



@timed
def main ():
    a = a0
    b = b0
    J = []
    iterations = 0
    last_J = 0

    while True:
        print(iterations , a , b , last_J)

        a , b = gradient_descent(yearDf , priceDf , a , b , alpha)
        iterations += 1

        last_J = cost(yearDf , priceDf , a , b)
        J.append(last_J)

        if np.isclose(last_J , 0 , atol=1e-04) or iterations == max_iterations:
            break


    print("a={} , b={} with a0={} , b0={} , alpha={} in {} iterations".format(a , b , a0 , b0 , alpha , iterations))

    return J


J = main()

plt.plot(J , "b.")
plt.show()

try:
    quit()
except:
    print("Bye!")