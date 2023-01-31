
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
import requests
path='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv'
res = requests.get(path)

print(res.status_code)

if res.status_code == 200:
    with open("FuelConsumption.csv", "wb") as f:
        f.write(res.content)
        
df = pd.read_csv("FuelConsumption.csv")
#print(df.head(10))

cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
# print(cdf.head(9))
# plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='blue')
# plt.xlabel("Engine size")
# plt.ylabel("Emission")
# plt.show()

# plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color='blue')
# plt.xlabel("Number of Cylinders")
# plt.ylabel("Emission")
# plt.show()

msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

# plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
# plt.xlabel("Engine size")
# plt.ylabel("Emission")
# plt.show()
# simple regression model
# from sklearn import linear_model
# regr = linear_model.LinearRegression()
# train_x = np.asanyarray(train[['ENGINESIZE']])
# train_y = np.asanyarray(train[['CO2EMISSIONS']])
# regr.fit(train_x, train_y)
# the coefficients
# print('coefficients: ', regr.coef_)
# print('Intercept: ', regr.intercept_)

# plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
# plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
# plt.xlabel("Engine size")
# plt.ylabel("Emission")
# plt.show()

# multiple regression model

from sklearn import linear_model
regr = linear_model.LinearRegression()
x = np.asanyarray(train[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY']])
y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(x, y)
#The coeffiencts
print('Coefficients: ', regr.coef_)

#Prediction
y_hat = regr.predict(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY']])
x = np.asanyarray(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY']])
y = np.asanyarray(test[['CO2EMISSIONS']])
print("Residual sum of squares: %.2f" % np.mean((y_hat - y)**2))
print('Variance score: %.2f' % regr.score(x, y))