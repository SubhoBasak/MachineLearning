import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('houseprices.csv', usecols = ['area', 'bedrooms', 'price'])

x = df[['area', 'bedrooms']]
y = df['price']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

model = LinearRegression()
model.fit(x_train, y_train)
print('Accuracy : ', model.score(x_test, y_test))

x_area_prd = int(input('Enter the area of the house : '))
x_bedrooms_prd = int(input('Enter the number of bedrooms of the house : '))
x_prd = np.array([[x_area_prd, x_bedrooms_prd]])

y_prd = model.predict(x_prd)
print('The house price will be about : ', y_prd)
