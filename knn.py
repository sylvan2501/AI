import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import preprocessing
import requests
path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv"
res = requests.get(path)

print(res.status_code)

if res.status_code == 200:
    with open("teleCust1000t.csv", "wb") as f:
        f.write(res.content)

df = pd.read_csv('teleCust1000t.csv')
#print(df)
# print(df['custcat'].value_counts())
# df.hist(column='income', bins=50)
# plt.show()
# print(df.columns)
# print(df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']].values)
X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']].values
# print(X[0:5])
y = df['custcat']
# print(y[0:5])
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

#print(X[0:5])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
# print('Train set:',X_train.shape, y_train.shape)
# print('Test set:', X_test.shape, y_test.shape)


# K nearest neighbor (KNN)
from sklearn.neighbors import KNeighborsClassifier

k = 6
neigh = KNeighborsClassifier(n_neighbors= k).fit(X_train, y_train)
yhat = neigh.predict(X_test)
print(yhat[0:5])

from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))

Ks = 10

mean_acc = np.zeros((Ks - 1))
std_acc = np.zeros((Ks - 1))

for n in range(1, Ks):
    #Train Model and Predict
    neigh = KNeighborsClassifier(n_neighbors= n).fit(X_train, y_train)
    yhat = neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)
    
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

mean_acc

plt.plot(range(1, Ks), mean_acc, 'g')
plt.fill_between(range(1, Ks), mean_acc - 1 * std_acc, mean_acc + 1 * std_acc, alpha=0.10)
plt.fill_between(range(1, Ks), mean_acc - 3 * std_acc, mean_acc + 3 * std_acc, alpha=0.10, color='green')
plt.legend(('Accuracy ','+/- 1xstd', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K')
plt.tight_layout()
plt.show()