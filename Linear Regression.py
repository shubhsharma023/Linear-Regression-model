# importing all the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# deciding size of the plot
plt.figure(figsize=[9, 7])

# taking data file/csv file
data = pd.read_csv('cars_100.csv')

# taking column 8 of the data set and assigning that column variable X which is mileage of the car
X = data.iloc[:, 8]

# taking column 9 of the data set and assigning that column variable Y which is the engine CC
Y = data.iloc[:, 9]
# So their linear regression will give the correlation with the CC and mileage of engine which shows efficient built of the engine in the present market
# finding mean of the data using numpy
X_mean = np.mean(X)
Y_mean = np.mean(Y)

# initializing the values from 0 to range X
num = 0
den = 0
for i in range(len(X)):
    num += (X[i] - X_mean)*(Y[i] - Y_mean)
    den += (X[i] - X_mean)**2

x = X-X_mean
y = Y-Y_mean
xy = x*y
sq_x = x**2
sq_y = y**2
sum_of_squares_x = sq_x.sum()
sum_of_squares_y = sq_y.sum()
sum_of_xy = xy.sum()

# finding the slope m
m = num / den

# finding the intercept c
c = Y_mean - (m*X_mean)

# predicting the slope
Y_pred = m*X + c

# Finding Karl Pearson's coefficient of correlation
r = (sum_of_xy)/((sum_of_squares_x**0.5)*(sum_of_squares_y**0.5))


# printing the required data
print("m =", m, "c =", c)
print("Predicted Y = ", Y_pred)

print("Karl Pearson's coefficient of correlation = ", r)
print('Since the slope of the graph is decreasing so they are negatively correlated.')

# actual scatter
plt.scatter(X, Y)
# predicted line
plt.plot(X, Y_pred, color='red', label='Regression Line')

plt.scatter(X, Y, color='green', label='Scatter Plot')
plt.title('Scatter map of data of engine CC and torque produced by the engine of 100 cars present in market.')
plt.xlabel('MILEAGE')
plt.ylabel('ENGINE (CC)')
plt.legend(loc='best', shadow=True)
plt.show()
