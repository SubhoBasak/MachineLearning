import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('insurance_data.csv', delimiter = '\t')

x = df['age'].values.reshape(-1, 1)
y = df['bought_insurance'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

model = LogisticRegression(solver = 'lbfgs')
model.fit(x_train, y_train)

print('Accuracy : ', model.score(x_test, y_test))
