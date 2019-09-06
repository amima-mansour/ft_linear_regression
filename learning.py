#!/Users/amansour/.brew/Cellar/python/3.7.4/bin/python3.7
import pandas as pd
import matplotlib.pyplot as plt
import math


def init_plot():
    # Configurate the plot
    # Say, "the default sans-serif font is COMIC SANS"
    plt.rcParams['font.sans-serif'] = "Comic Sans MS"
    # Then, "ALWAYS use sans-serif fonts"
    plt.rcParams['font.family'] = "sans-serif"
    plt.title("Price of cars based on their metrage", fontsize = 20, y=1.02, color = 'green', fontweight = 'bold')
    plt.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1)
    plt.xlabel("Km", fontsize=14, fontweight = 'bold')
    plt.ylabel("Price", fontsize=14, fontweight = 'bold')

def cost_function(data, theta0, theta1):
    cost = 0.0
    for index, row in data.iterrows():
        y = row['price']
        x = row['km']
        cost = cost + (theta0 + x * theta1 - y) ** 2
    cost /= 2 * data['km'].size
    return (cost)

def learning(data):
    iterations = 1
    size = data['km'].size
    cost = -1 # we put a negative number because cost can not not be negative
    learning_rate = 0.01 # a checker
    cost_history = []
    theta1, theta0 = 0, 0
    for i in range(iterations):
        tmp_theta1 = 0.0
        tmp_theta0 =  0.0
        for index, row in data.iterrows():
            y = row['price']
            x = row['km']
            tmp_theta1 = tmp_theta1 + (theta1 * x + theta0 - y) * x
            tmp_theta0 =  tmp_theta0 + (theta1 * x + theta0 - y)
        print( 0.01 * tmp_theta1 / size)
        theta0 -= 2 *  (tmp_theta0 / size) * learning_rate
        theta1 -= 2 * (tmp_theta1 / size) * learning_rate
        #print(theta0, theta1)
        cost = cost_function(data, theta0, theta1)
        cost_history.append(cost)
        #if cost < 0 or tmp_cost < cost:
        #    final_theta0 = theta0
         #   final_theta1 = theta1
        #  cost = tmp_cost
        # Log Progress
        #if i % 10 == 0:
        print("iter={}    tetha1={:.2f}    theta0={:.4f}    cost={:.2}".format(i, theta1, theta0, cost))
    return theta0, theta1

if __name__ == '__main__':
    # Read data from file 'filename.csv'
    data = pd.read_csv("data_test.csv")
    # Learning
    theta0, theta1 = learning(data)
    # print(theta0, theta1)
    # Plot Data frame
    init_plot()
    data.plot(x ='km', y='price', label='Price Per Km',kind = 'scatter', color='blue')
    plt.plot(data['km'], theta1 * data['km'] + theta0 , '-r')  # solid green
    #plt.show()