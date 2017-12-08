print(__doc__)


# => PLOT THE TENDANCY OF THE COUNT ON EACH DAY 
# TIME (INDEX) * COUNT (NUMBER OF BIKE)


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import sklearn
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
# from sklearn import datasets

# Load the data from Lottery.csv
# filename = 'spam.csv'
filename_day = 'day.csv'
filename_hour = 'hour.csv'
mydata_day = pd.read_csv(filename_day)
mydata_hour = pd.read_csv(filename_hour)

md = pd.DataFrame(mydata_day)
md.head()

# target data
# md.cnt[:5]
md.cnt

# Plot a figure of bike counts throw time (x = index)
md.plot(y='cnt',use_index=True)


# Use all [13] parameters to fit a linear regression  model.  
# Two other parameters that you can pass to linear regression object are fit_intercept and normalize.
from sklearn.linear_model import LinearRegression
X = md.drop('cnt',axis=1)
# Drop date day because only takes float values
X = X.drop('dteday',axis=1)

lm = LinearRegression()
lm

lm.fit(X,md.cnt)

# Print the intercept and number of coefficients.
print 'Estimated intercept coefficients: ', lm.intercept_
print 'Number of coefficients: ',len(lm.coef_)

#  construct a data frame that contains features and estimated coefficients.
pd.DataFrame(zip(X.columns, lm.coef_),columns = ['features','estimatedCoefficients'])

# See that high correlation between casual, registered and cnt=> should drop them (they are part of the prediciton too)
X = X.drop('casual',axis=1)
X = X.drop('registered',axis=1)

# Do the coef again
lm = LinearRegression()
lm.fit(X,md.cnt)

# Print the intercept and number of coefficients.
print 'Estimated intercept coefficients: ', lm.intercept_
print 'Number of coefficients: ',len(lm.coef_)

#  construct a data frame that contains features and estimated coefficients.
pd.DataFrame(zip(X.columns, lm.coef_),columns = ['features','estimatedCoefficients'])

# See that high correlation between year and cnt
# Plot a scatter plot between True cnt and True year
# Not really interesting : only year 1 and year 2
plt.scatter(X.yr,md.cnt)
plt.xlabel("Year")
plt.ylabel("Number of bikes")
plt.title("Relationship between year and bike")
plt.show()

# Calculate the predicted cnt (Y^i)  using lm.predict. 
# Display the first 5 cnt
lm.predict(X)[0:5]
# => array([ 2005.65455703,  1474.36497022,  1609.08275066,  1829.95866891, 2088.33068989])

# Plot a scatter plot to compare true cnt and the predicted cnt.
plt.scatter(md.cnt,lm.predict(X))
plt.xlabel("Number of bikes: $cnt_i$")
plt.ylabel("Predicted number of bikes: $\hat{cnt}_i$")
plt.title("Number of bikes vs Predicted number of bikes: $cnt_i$ vs $\hat{cnt}_i$")
plt.show()

# Can see that there is some errors
# Calculation of the mean squared error
mseFull = np.mean((md.cnt - lm.predict(X))**2)
# mseFull = np.median((md.cnt - lm.predict(X))**2)
print mseFull
# Answer is mseFull = 743,646.509994


# === PART 2 ====

# Try to compute the linear regression of only one feature
lm = LinearRegression()
lm.fit(X[[hum]],md.cnt)
mseHUM = np.mean((md.cnt - lm.predict(X[['hum']]))**2)
print mseHUM
# Error is worst: mseHUM = 3709682.6527
# A single feature is not a good predictor of housing prices.

# === PART 3 : training and testing dataset ====

X_train = X[:-50]
X_test = X[-50:]
Y_train = md.cnt[:-50]
Y_test = md.cnt[-50:]

print X_train.shape
print X_test.shape
print Y_train.shape
print Y_test.shape


# Train-test split:
# Divide your data sets randomly. Scikit learn provides a function called train_test_split to do this
# X_train, X_test, Y_train, Y_test = sklearn.cross_validation.train_test_split(X, md.cnt, test_size=0.33,random_state=5)
X_train, X_test, Y_train, Y_test = train_test_split(X, md.cnt, test_size=0.33,random_state=5)
print X_train.shape
print X_test.shape
print Y_train.shape
print Y_test.shape

# Using linear regression on the train-test data sets
lm = LinearRegression()
lm.fit(X_train,Y_train)
pred_train = lm.predict(X_train)
pred_test = lm.predict(X_test)

# Calculate mean squared error for training and test data
print 'Fit a model X_train, and calculate MSE with Y_train:', np.mean((Y_train - lm.predict(X_train)) ** 2)
# Output is Fit a model X_train, and calculate MSE with Y_train: 693,204.044608
print 'Fit a model X_train, and calculate MSE with X_test, Y_test:', np.mean((Y_test - lm.predict(X_test)) ** 2)
# Output is Fit a model X_train, and calculate MSE with X_test, Y_test: 886,811.179782

#  === PART 4 : residual plots (to visualize the error in the data) ===
plt.scatter(lm.predict(X_train),lm.predict(X_train) - Y_train, c='b', s=40, alpha=0.5)
plt.scatter(lm.predict(X_test),lm.predict(X_test) - Y_test, c='g', s=40)
plt.hlines(y=0, xmin=0, xmax=8000)
plt.title('Residual Plot using training (blue) and test (green) data')
plt.ylabel('Residuals')
plt.show()

# If you have done a good job then your data should be randomly scattered around line zero. 
# If you see structure in your data, that means your model is not capturing some thing. 
# Maye be there is a interaction between 2 variables that you are not considering, or may be you are measuring time dependent data. 
# If you get some structure in your data, you should go back to your model and check whether you are doing a good job with your parameters.


# ==== OTHER MODELS (SVR RBF + SVR LINEAR) ===
from sklearn.svm import SVR

# svr_rbf = SVR(kernel = 'rbf', C=1e3, gamma=0.1)
# y_rbf = svr_rbf.fit(X_train, Y_train).predict(X_train)
# svr_lin = SVR(kernel = 'linear', C=1e3)
# y_lin = svr_lin.fit(X_train, Y_train).predict(X_train)
# svr_poly = SVR(kernel = 'poly', C=1e3, degree=4)
# y_poly = svr_poly.fit(X_train, Y_train).predict(X_train)
y=md.cnt

svr_rbf = SVR(kernel = 'rbf')
y_rbf = svr_rbf.fit(X, y).predict(X)
svr_lin = SVR(kernel = 'linear', C=1e3)
y_lin = svr_lin.fit(X, y).predict(X)
# svr_poly = SVR(kernel = 'poly', C=1e3, degree=4)
# y_poly = svr_poly.fit(X_train, Y_train)


# plt.plot(range(0,len(X_train),1),Y_train,color='g',lw=2)
# plt.plot(range(0,len(X_test),1),Y_test,color='g',lw=2)
# plt.plot(range(0,len(X),1),md.cnt,color='g',lw=2)
plt.plot(range(0,len(X),1),y_rbf,color='g',lw=2)
plt.plot(range(0,len(X),1),y_lin,color='b',lw=2)
# plt.plot(X_train.index,y_poly,color='r',lw=2)
plt.show()


# # Plot a scatter plot to compare true cnt and the predicted cnt.
# plt.scatter(md.cnt,y_lin)
# plt.xlabel("Number of bikes: $cnt_i$")
# plt.ylabel("Predicted number of bikes: $\hat{cnt}_i$")
# plt.title("Number of bikes vs Predicted number of bikes: $cnt_i$ vs $\hat{cnt}_i$")
# plt.show()


# svr_lin.fit(X,y).predict(X)[:5]

# plt.show()


# OLS => Ordinary least square : try to minimize the error. 
# + another one 





