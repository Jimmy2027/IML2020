import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error


sample = pd.read_csv('sample.csv')
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
y = train['y'].to_list()
y_train = train.iloc[:, 1].values
x_train = train.iloc[:, 2:].values
x_test = test.iloc[:, 1:].values


model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
# mse = mean_squared_error(y_pred, y_test)
# print(mse)

plt.figure()
plt.scatter(x_test[:, 1], y_pred)
plt.show()

Id = [x for x in range(10000, len(y_pred) + 10000)]

df_solution = pd.DataFrame({'Id': Id, 'y': y_pred})

df_solution.to_csv('solution.csv', index=False)
