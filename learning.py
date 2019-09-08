#!/Users/amansour/.brew/Cellar/python/3.7.4/bin/python3.7
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def display_plot(data, x, y, cost_history):
    # Configurate the plot
    fig, axs = plt.subplots(2, figsize=(8,8))
    # Say, "the default sans-serif font is COMIC SANS"
    plt.rcParams['font.sans-serif'] = "Comic Sans MS"
    # Then, "ALWAYS use sans-serif fonts"
    plt.rcParams['font.family'] = "sans-serif"
    axs[0].scatter(data[:, 0], data[:, 1], color='blue', label="data")
    axs[0].plot(x, y, '-r', label="line")
    axs[0].set_title("Price of cars based on their metrage", fontsize = 15, y=1.02, color = 'green', fontweight = 'bold')
    axs[0].legend(fancybox=True, framealpha=1, shadow=True, borderpad=1)
    axs[0].set_xlabel("Km", fontsize=14, fontweight = 'bold')
    axs[0].set_ylabel("Price", fontsize=14, fontweight = 'bold')
    axs[1].plot(np.arange(10) * 10, cost_history, '-r', label="cost")
    axs[1].set_title("Cost per iteration", fontsize=15, y=1.02, color='green', fontweight='bold')
    axs[1].legend(fancybox=True, framealpha=1, shadow=True, borderpad=1)
    axs[1].set_xlabel("Iteration", fontsize=14, fontweight='bold')
    axs[1].set_ylabel("Cost", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def cost_function(theta, x, y, size):
    cost = (theta[0] + x * theta[1] - y) ** 2
    cost /= 2 * size
    return(np.sum(cost))

def scale_data(x):
	return (x - np.mean(x)) / np.std(x)

def descale_data(x, x_ref):
	return x * np.std(x_ref) + np.mean(x_ref)

def predict(x, theta):
    return theta[0] + theta[1] * x

def learning(x, y, size, alpha, iterations):
    theta = np.zeros(2)
    cost_history = []
    for i in range(0, iterations):
        tmp_theta = np.zeros(2)
        for j in range(0, size):
            tmp_theta[0] += predict(x[j], theta) - y[j]
            tmp_theta[1] += ((predict(x[j], theta) - y[j]) * x[j])
        theta[0] -= (tmp_theta[0] * alpha) / size
        theta[1] -= (tmp_theta[1] * alpha) / size
        if i % 10 == 0:
            cost_history.append(cost_function(theta, x, y, size))
    return theta, cost_history

if __name__ == '__main__':
    # Read data from file 'filename.csv'
    data = pd.read_csv("data.csv")
    # convert dataframe to matrix
    data = np.array(data)
    size = np.shape(data)[0]
    # # plt.scatter(data[:, 0], data[:, 1], color='blue')
    # Learning
    iterations = 100
    learning_rate = 0.1
    x = scale_data(data[:, 0])
    y = scale_data(data[:, 1])
    theta, cost_history =  learning(x, y, size, learning_rate, iterations)
    # use data
    y = predict(x, theta)
    x = descale_data(x, data[:, 0])
    y = descale_data(y, data[:, 1])
    # Plot
    display_plot(data, x, y, cost_history)