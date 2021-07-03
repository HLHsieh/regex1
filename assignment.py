from sklearn.linear_model import LinearRegression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("regrex1.csv")
df

plt.scatter(df.x, df.y)
plt.show()

linear_regressor = LinearRegression()

X = df.iloc[:, 1].values.reshape(-1, 1)
Y = df.iloc[:, 0].values.reshape(-1, 1)

linear_regressor.fit(X, Y)

Y_pred = linear_regressor.predict(X)
Y_pred


plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()
