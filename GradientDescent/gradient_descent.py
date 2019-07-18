import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x = [1, 3, 2, 5, 3, 4, 6, 2]
y = [2, 2, 1, 4, 3, 5, 6, 3]
m = b =0
lrn = 0.1
line = lambda X: m*X+b
#line = lambda X: model.coef_*X+model.intercept_

def graph(func, x_range):
    x_points = np.array(x_range).reshape(-1, 1)
    y_points = eval(func)

    plt.scatter(x, y)
    plt.plot(x_points, y_points)
    plt.show()

def plot_line(func, x_points):
    x_points = range(int(min(x_points))-1, int(max(x_points))+1)
    y_points = [func(i) for i in x_points]
    plt.scatter(x, y)
    plt.plot(x_points, y_points)
    plt.show()

def summetion(func, x_points, y_points):
    t1 = t2 = 0
    for i in range(len(x_points)):
        t1 = func(x_points[i])- y_points[i]
        t2 = t1*x_points[i]

    return t1/len(x_points), t2/len(x_points)

model = LinearRegression()
model.fit(np.array(x).reshape(-1, 1), y)

#graph('model.coef_*x_points+model.intercept_', range(0, 8))
#plot_line(line, x)

for i in range(50):
    s1, s2 = summetion(line, x, y)
    m = m- lrn*s1
    b = b- lrn*s2
plot_line(line, x)
