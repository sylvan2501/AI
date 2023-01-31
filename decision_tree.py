import pandas as pd
import pylab as pl
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree
import requests

path= 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv'
res = requests.get(path)


if res.status_code == 200:
    with open("drug200.csv", "wb") as f:
        f.write(res.content)

my_data = pd.read_csv('drug200.csv', delimiter=',')
print(my_data.shape) #this is the size of the data
X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
print(X[0:5])

from sklearn import preprocessing
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F', 'M'])
X[:,1] = le_sex.transform(X[:,1])

le_BP = preprocessing.LabelEncoder()
le_BP.fit(['LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])

le_Chol = preprocessing.LabelEncoder()
le_Chol.fit(['NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3])
print(X[0:5])

y = my_data['Drug']
print(y[0:5])

from sklearn.model_selection import train_test_split
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)

print('Shape of X training set {}'.format(X_trainset.shape))
print('Size of Y training set {}'.format(y_trainset.shape))
print('Shape of X testing set {}'.format(X_testset.shape))
print('Size of Y testing set {}'.format(y_testset.shape))
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth=4)
drugTree.fit(X_trainset, y_trainset)
predTree = drugTree.predict(X_testset)

from sklearn import metrics
import matplotlib.pyplot as plt
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))

tree.plot_tree(drugTree)
plt.show()

