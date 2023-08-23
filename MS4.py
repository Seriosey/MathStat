import numpy as np
from scipy.stats import kendalltau, poisson, f_oneway, ttest_ind, f, chi2
from statsmodels.stats.proportion import proportions_ztest

# Создание связных выборок Х и Y
np.random.seed(1)
X = np.random.normal(5, 8, 100)
epsilon = np.random.normal(0, 1, 100)
Y = 3*X - 4 + 0.1*epsilon

# Вычисление коэффициента корреляции Тау Кендалла
tau, pt = kendalltau(X, Y)
print("Значение коэффициента корреляции Тау Кендалла:", tau)
#print("Значение p-значения:", pt)
X_modified = np.concatenate((X[:50], Y[50:]))

tau, ptm = kendalltau(X_modified, Y)
print("Измененное Значение коэффициента корреляции Тау Кендалла:", tau, '\n')
#print("Значение p-значения:", ptm)


# Создание несвязных выборок X1 и X2
np.random.seed(1)
X1 = np.random.normal(5, 8, 100)
X2 = np.random.normal(5, 12, 100)

# Проверка гипотезы о равенстве средних
t_stat, p_value = ttest_ind(X1, X2)
f_stat5, p_value5 = f_oneway(X1, X2)
print("Значение статистики t для среднего:", t_stat)
print("Значение p-значения:", p_value)
print("Значение статистики F для среднего:", f_stat5)
print("Значение p-значения:", p_value5, '\n')

f0 = np.var(X1, ddof=1)/np.var(X2, ddof=1) #calculate F test statistic 
dfn = X1.size-1 #define degrees of freedom numerator 
dfd = X2.size-1 #define degrees of freedom denominator 
p = 1-f.cdf(f0, dfn, dfd) #find p-value of F test statistic 
print("Новое Значение статистики F для дисперсии:", f0)
print("Значение p-значения:", p, '\n')

# Создание несвязных выборок X и Y
np.random.seed(1)
X = np.random.normal(5, 8, 100)
Y = np.random.normal(3, 5, 100)

# Проверка критерия Фишера на равенство дисперсий
t_stat3, p_value = ttest_ind(X, Y)
f_stat, p_value1 = f_oneway(X, Y)
print("Значение статистики t для среднего:", t_stat3)
print("Значение p-значения:", p_value)
print("Значение статистики F для среднего:", f_stat)
print("Значение p-значения:", p_value1, '\n')
f35 = np.var(X, ddof=1)/np.var(Y, ddof=1) #calculate F test statistic 
dfn = X.size-1 #define degrees of freedom numerator 
dfd = Y.size-1 #define degrees of freedom denominator 
p = 1-f.cdf(f35, dfn, dfd) #find p-value of F test statistic 
print("Новое Значение 3,5 статистики F для дисперсии:", f35)
print("Значение p-значения:", p, '\n')
X_mod = np.concatenate((X[:50], Y[50:]))
f35m = np.var(X_mod, ddof=1)/np.var(Y, ddof=1) #calculate F test statistic 
dfn = X_mod.size-1 #define degrees of freedom numerator 
dfd = Y.size-1 #define degrees of freedom denominator 
pm = 1-f.cdf(f35m, dfn, dfd) #find p-value of F test statistic 
print("Новое Значение 3,5 mod статистики F для дисперсии:", f35m)
print("Значение p-значения:", pm, '\n')




np.random.seed(1)
X1 = np.random.poisson(4, 100)
X2 = np.random.poisson(5, 100)
X_mod = np.concatenate((X1[:50], X2[50:]))

# Проверка асимптотического критерия равенства параметров λ
z_stat, p_value = ttest_ind(X_mod, X2)
print("Значение статистики Z Пуассона:", z_stat)
print("Значение p-значения:", p_value, '\n')

lambda_null = (X_mod.mean() + X2.mean()) / 2
lambda_alt = [X_mod.mean(), X2.mean()]

# Рассчет значения статистики отношения правдоподобия
ll_null = np.sum(poisson.logpmf(X_mod, lambda_null)) + np.sum(poisson.logpmf(X2, lambda_null))
ll_alt = np.sum(poisson.logpmf(X_mod, lambda_alt[0])) + np.sum(poisson.logpmf(X2, lambda_alt[1]))
likelihood_ratio = -2 * (ll_null - ll_alt)

# Рассчет p-значения
p_value = chi2.sf(likelihood_ratio, 1)

print("Значение статистики отношения правдоподобия для Пуассона:", likelihood_ratio)
print("Значение p-значения:", p_value)





# Создание несвязных выборок X1 и X2
np.random.seed(1)
X1 = np.random.binomial(1, 0.25, 100)
X2 = np.random.binomial(1, 0.4, 100)
X_mod = np.concatenate((X1[:50], X2[50:]))

# Проверка асимптотического критерия равенства вероятностей успеха
count = np.array([X1.sum(), X2.sum()])
nobs = np.array([len(X1), len(X2)])
z_stat, p_value = proportions_ztest(count, nobs)
print("Значение статистики Z:", z_stat)
print("Значение p-значения:", p_value, '\n')
count = np.array([X_mod.sum(), X2.sum()])
nobs = np.array([len(X_mod), len(X2)])
z_stat, p_value = proportions_ztest(count, nobs)
print("Значение статистики Z:", z_stat)
print("Значение p-значения:", p_value)