#!/Users/amansour/.brew/Cellar/python/3.7.4/bin/python3.7
import predicting as predict
import pandas as pd
import numpy as np

def predict_function(theta0, theta1):
    mileage = input("Please enter a mileage: ")
    try:
        mileage = float(mileage)
        print(theta0 + theta1 * mileage)
        print("Estimated price for {} Km is {:.3f}".format(mileage, theta0 + theta1 * mileage))
    except ValueError as error:
        print(error)
        print("error: Please enter a number.")
        predict_function(theta0, theta1)

if __name__ == '__main__':
    theta = [0, 0]
    try:
        # Read data from file 'filename.csv'
        theta = np.loadtxt("theta.txt", dtype=np.longdouble)
        print(theta)
    except:
        print("Learning is not done yet")
    predict_function(theta[0], theta[1])