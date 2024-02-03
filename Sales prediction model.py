#This program is a part of Fundementals of Machine learning course NYU Paris
# Coded by Mahmoud-K-Ismail
# This is a Machine learning model that predicts the Sales based on 3 Ad features(TV, Radio, Newspaper )
# it also decided which features shall be considered

# imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# read data into a DataFrame
data = pd.read_csv('data.csv', index_col=0 )

print (data.shape) #there are 200 examples in this dataset
data.describe()

data['Sales'].hist(bins = 20)

data.boxplot()

data.plot(kind= 'scatter',x= 'TV', y = 'Sales')
data.plot(kind= 'scatter',x= 'Radio', y = 'Sales')
data.plot(kind= 'scatter',x= 'Newspaper', y = 'Sales')

data.corr()
sns.heatmap(data.corr(),annot=True)

columns = ['TV']

ls = LinearRegression()
ls.fit(data[columns],data['Sales'])
ls.intercept_,ls.coef_[0]

# manually calculate the prediction
x = 50_000
b0 = 7.0325935491276885
b1 = 0.04753664043301979
y = b0+ b1*x
print(y)

the_value = 50_000
New_data = pd.DataFrame({'TV' : [the_value]})
ls.predict(New_data)[0]

dt = pd.DataFrame(data['TV'])
plt.plot(dt,ls.predict(dt))
plt.scatter(dt,data['Sales'],color = 'red')


columns2 = ['TV','Radio','Newspaper']
ls2 = LinearRegression()
ls2.fit(data[columns2],data['Sales'])
ls2.intercept_,ls2.coef_


e = data['Sales']-ls2.predict(data[['TV','Radio','Newspaper']])
e.hist(bins =20)


def confidence_interval(X,y,a,k):
  assert(a<1)
  (N,d) = X.shape
  lsf = LinearRegression()
  lsf.fit(X,y)
  variance = np.linalg.norm(y.values-lsf.predict(X))**2/(N-d-1)
  Phi = np.concatenate((np.reshape(np.ones(N),(N,1)),X.to_numpy()),axis=1)
  s = np.diag(np.linalg.inv(np.dot(Phi.T,Phi)))[k]
  q = stats.t(df=N-d-1).ppf(1-(1-a)/2)
  return (lsf.coef_[k-1]-q*np.sqrt(variance*s/N),lsf.coef_[k-1]+q*np.sqrt(variance*s/N))

confidence_interval_tv = confidence_interval(data[['TV']],data[['Sales']], 0.95, 1)
confidence_interval_newspaper = confidence_interval(data[['Newspaper']],data[['Sales']], 0.95, 1)
confidence_interval_radio = confidence_interval(data[['Radio']],data[['Sales']], 0.95, 1)

# Print the intervals without array notation
print(f"Confidence interval for TV is ({float (confidence_interval_tv[0])}, {float (confidence_interval_tv[1])})")
print(f"Confidence interval for Newspaper is ({confidence_interval_newspaper[0]}, {confidence_interval_newspaper[1]})")
print(f"Confidence interval for Radio is ({confidence_interval_radio[0]}, {confidence_interval_radio[1]})")

print(f"confidence interval for TV is {confidence_interval(data[['TV']],data[['Sales']],0.99,1)}")
print (f"confidence interval for Newspaper is {confidence_interval(data[['Newspaper']],data[['Sales']],0.99,1)}")
print (f"confidence interval for Newspaper is {confidence_interval(data[['Radio']],data[['Sales']],0.99,1)} ")

X = data[columns2]
y = data['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
ls.fit(X_train,y_train)
print(ls.coef_)
ls2.fit(X_train[['TV','Radio']],y_train)
print(ls2.coef_)

y1_predict = ls.predict(X_test)
y2_predict = ls2.predict(X_test[['TV','Radio']])
ms1 = mean_squared_error(y_test, y1_predict)
ms2 = mean_squared_error(y_test, y2_predict)
print (f"The mean square error of the 3 features together is : {ms1}")
print (f"The mean square error of the 2 features without considering newspapaprs is: {ms2}")

if (ms1 < ms2):
  print ("Model 1 (with the 3 features) shall be used")
else:
  print ("Model 2 (without the newspaper feature) shall be used")