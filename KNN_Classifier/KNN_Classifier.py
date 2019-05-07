import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv('breast-cancer-wisconsin.data.txt', header = None, na_values = '?')
df.fillna('99999', inplace = True)
df.drop(0, 1, inplace = True)

x = df.drop(10, 1)
y = df[10]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4)

model = KNeighborsClassifier()
model.fit(x_train, y_train)

print('Accuracy : ', model.score(x_test, y_test))
