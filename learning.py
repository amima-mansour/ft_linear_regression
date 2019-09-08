#!/Users/amansour/.brew/Cellar/python/3.7.4/bin/python3.7
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import math


def init_plot():
    # Configurate the plot
    # Say, "the default sans-serif font is COMIC SANS"
    plt.rcParams['font.sans-serif'] = "Comic Sans MS"
    # Then, "ALWAYS use sans-serif fonts"
    plt.rcParams['font.family'] = "sans-serif"
    # plt.title("Price of cars based on their metrage", fontsize = 20, y=1.02, color = 'green', fontweight = 'bold')
    # plt.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1)
    # plt.xlabel("Km", fontsize=14, fontweight = 'bold')
    # plt.ylabel("Price", fontsize=14, fontweight = 'bold')

def cost_function(data, theta0, theta1):
    cost = 0.0
    size = np.shape(data)[0]
    for el in data:
        y = el[0]
        x = el[1]
        cost = cost + (theta0 + x * theta1 - y) ** 2
    cost /= 2 * size
    return (cost)

def scale_data(x):
	return (x - np.mean(x)) / np.std(x)

def descale_data(x, x_ref):
	return x * np.std(x_ref) + np.mean(x_ref)

def predict(x, theta):
    return theta[0] + theta[1] * x

def estimate_price(theta_0, theta_1, x):
	return theta_0 + theta_1 * x

def compute_gradients(x, y, m, alpha, iterations):
	theta = np.zeros((1, 2))
	for i in range(0, iterations):
		tmp_theta = np.zeros((1, 2))
		for j in range(0, m):
			tmp_theta[0, 0] += (estimate_price(theta[0, 0], theta[0, 1], x[j]) - y[j])
			tmp_theta[0, 1] += ((estimate_price(theta[0, 0], theta[0, 1], x[j]) - y[j]) * x[j])
		theta -= (tmp_theta * alpha) / m
	return theta

def learning(x, y, size, alpha, iterations):
    theta = np.zeros(2)
    for i in range(0, iterations):
        tmp_theta = np.zeros(2)
        for j in range(0, size):
            tmp_theta[0] += predict(x[j], theta) - y[j]
            tmp_theta[1] += ((predict(x[j], theta) - y[j]) * x[j])
            print(tmp_theta[0])
        theta[0] -= (tmp_theta[0] * alpha) / size
        theta[1] = (tmp_theta[1] * alpha) / size
        # print(theta[0], theta[1])
    return theta

if __name__ == '__main__':
    # Read data from file 'filename.csv'
    data = pd.read_csv("data.csv")
    # plot scatter
    init_plot()
    data.plot(x='km', y='price', label='Price Per Km', kind='scatter', color='blue')
    # convert dataframe to matrix
    data = np.array(data)
    size = np.shape(data)[0]
    # # plt.scatter(data[:, 0], data[:, 1], color='blue')
    # Learning
    iterations = 200
    learning_rate = 0.3  # a checker
    x = scale_data(data[:, 0])
    y = scale_data(data[:, 1])
    theta =  compute_gradients(x, y, size, learning_rate, iterations)
    theta[0, 0] = -9.39935521e-18
    print(theta)

    # use data
    y = estimate_price(theta[0,0],theta[0,1], x)
    x = descale_data(x, data[:, 0])
    y = descale_data(x, data[:, 1])
    print(y)
    # Plot Data frame
    plt.plot(x, y , '-r')
    # plt.show()