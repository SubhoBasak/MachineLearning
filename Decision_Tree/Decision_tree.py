import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv('salaries.csv')

company_le = LabelEncoder()
job_le = LabelEncoder()
degree_le = LabelEncoder()

df['company'] = company_le.fit_transform(df['company'])
df['job'] = job_le.fit_transform(df['job'])
df['degree'] = degree_le.fit_transform(df['degree'])

x = df.drop('salary_more_then_100k', 1).values
y = df['salary_more_then_100k'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

model = DecisionTreeClassifier()
model.fit(x_train, y_train)

print('Accuracy : ', model.score(x_test, y_test))
