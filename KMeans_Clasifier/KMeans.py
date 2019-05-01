import pandas as pd
import matplotlib.pyplot as plt
<<<<<<< HEAD
from sklearn.cluster import KMeans
=======
from sklearn.classifier import KMeans
>>>>>>> ccbc43a16f9c2dd50967df910ba6ebfcbb323c0d

df = pd.read_csv('xclara.csv')

x1 = df['V1']
x2 = df['V2']

<<<<<<< HEAD
range_cluster = range(1, 11)
sse = []

for k in range_cluster:
    model = KMeans(n_clusters = k)
    model.fit(df)
    sse.append(model.inertia_)

# Here we used Elbow method to determine the optimal number of clusters

plt.plot(range_cluster, sse, color = 'pink', linestyle = '--',
         lw = 2, marker = '*', markeredgecolor = 'pink',
         markerfacecolor = 'cyan', markersize = 12)
plt.xlabel('Number of Clusters')
plt.ylabel('Cost')
plt.show()

# after closed the plot window, enter the number of the clusters manually

k = int(input('\nNumber of Clusters : '))

model = KMeans(n_clusters = k)
model.fit(df)

plt.scatter(x1, x2, c = model.labels_)
plt.show()
=======
for i in range(10):
    model = KMeans(n_cluster = )
>>>>>>> ccbc43a16f9c2dd50967df910ba6ebfcbb323c0d
