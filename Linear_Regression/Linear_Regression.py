#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split



house_size = [1240, 1800, 1543, 1788, 2200, 1090, 800, 1470, 1650, 1400, 1800]
house_price = [540, 780, 700, 730, 1020, 370, 210, 600, 740, 480, 630]

x = np.array(house_size).reshape(-1, 1)
y = np.array(house_price)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)



model = LinearRegression()
model.fit(x_train, y_train)
accuracy = model.score(x_test, y_test)

print('Accuracy : ', accuracy)


x_range = range(800, 2300)

def graph(formula, x_range):
    x_points = np.array(x_range).reshape(-1, 1)
    y_points = eval(formula)

    plt.scatter(x_train, y_train, color = 'red')
    plt.plot(x_points, y_points, color = 'green')
    plt.xlabel('House Size')
    plt.ylabel('House Price')
    plt.xticks(range(700, 2400, 200))
    plt.yticks(range(200, 1060, 100))
    plt.show()

graph('model.coef_ * x_points + model.intercept_', x_range)

x_prd = int(input('Enter the house size, you want to predict : '))
y_prd = model.predict([[x_prd]])
print('House price will be about : ', y_prd)
