#!/Users/amansour/.brew/Cellar/python/3.7.4/bin/python3.7
import predicting as predict
import os

def predict_function(theta0, theta1):
    mileage = input("Please enter a mileage: ")
    try:
        mileage = float(mileage)
        print("Estimated price for {}Km is {}".format(mileage, theta0 + theta1 * mileage))
    except ValueError as error:
        print(error)
        print("error: Please enter a number.")
        predict_function(theta0, theta1)

if __name__ == '__main__':
    theta0, theta1 = 0, 0
    try:
        with open('theta.txt', 'r') as f:
            theta = f.read()
            theta = theta.split('\n')
            theta0 = float(theta[0])
            theta1 = float(theta[1])
    except:
        print("Learning is not done yet")
    predict_function(theta0, theta1)