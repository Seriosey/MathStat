# -*- coding: windows-1251 -*-
import numpy as np

np.random.seed(0)
N = 1000
X = np.random.normal(5, 8, N)
epsilon = np.random.uniform(-2, 3, 1000)
Y = X + 2 * np.roll(X, 1) #+ epsilon

from statsmodels.graphics.tsaplots import plot_acf

#plot_acf(X, lags=20)
#plot_acf(Y, lags=20)

X_modified = np.concatenate((X[:N//2], np.random.normal(5, 12, N//2)))

from scipy.stats import f_oneway

f_statistic, p_value = f_oneway(X, X_modified)
print(f_statistic, '\n', p_value)

from scipy.stats import chi2_contingency

contingency_table = np.array([[np.sum((X - np.mean(X))**2), np.sum((X_modified - np.mean(X_modified))**2)]])

chi2_statistic, p_value, _, _ = chi2_contingency(contingency_table)

print("Хи-квадрат статистика:", chi2_statistic)
print("p-значение:", p_value)


epsilon = np.random.normal(0, 1, 500)
X = np.random.normal(5, 8, 500)
Y = 3 * X - 8 + 3 * epsilon

from sklearn.linear_model import LinearRegression

# Негруппированный случай
regressor_ungrouped = LinearRegression()
regressor_ungrouped.fit(X.reshape(-1, 1), Y)

# Группированный случай
X_grouped = np.mean(X.reshape(100, 5), axis=1)
Y_grouped = np.mean(Y.reshape(100, 5), axis=1)
regressor_grouped = LinearRegression()
regressor_grouped.fit(X_grouped.reshape(-1, 1), Y_grouped)


import matplotlib.pyplot as plt

plt.scatter(X, Y, label='Выборка')
#plt.plot(X, regressor_ungrouped.predict(X.reshape(-1, 1)), color='red', label='Линейная регрессия (негруппированный)')
plt.plot(X_grouped, regressor_grouped.predict(X_grouped.reshape(-1, 1)), color='green', label='Линейная регрессия (группированный)')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.show()