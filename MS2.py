import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

x = np.arange(-10, 10, 0.04)

# Генерация выборки объемом 500 из нормального распределения N(5, 8)
sample = np.random.normal(5, 8, 500)
grouped_sample, bins = np.histogram(sample, bins=100)

# Вычисление выборочного среднего и выборочного стандартного отклонения
sample_mean = np.mean(sample)
sample_std = np.std(sample, ddof=1)

g_sample_mean = np.mean(grouped_sample)
g_sample_std = np.std(grouped_sample, ddof=1)


confidence_interval_mean = stats.norm.interval(0.95, loc= sample_mean, scale= sample_std / np.sqrt(len(sample)))
my_interval = [sample_mean - 0.975*sample_std/(500**(1/2)), sample_mean + 0.975*sample_std/500**((1/2)) ]
assympt_interval = [sample_mean - 0.975*sample_std/(500**(1/2)), sample_mean + 0.975*sample_std/500**((1/2))]

# Доверительный интервал для дисперсии
confidence_interval_variance = ((len(sample) - 1) * sample_std ** 2 / stats.chi2.ppf(0.975, len(sample) - 1),
                               (len(sample) - 1) * sample_std ** 2 / stats.chi2.ppf(0.025, len(sample) - 1))



# Доверительный интервал для среднего
g_confidence_interval_mean = stats.norm.interval(0.95, loc= g_sample_mean, scale= g_sample_std / np.sqrt(len(grouped_sample)))
my_interval = [g_sample_mean - 0.975*g_sample_std/(500**(1/2)), g_sample_mean + 0.975*g_sample_std/500**((1/2)) ]
assympt_interval = [g_sample_mean - 0.975*g_sample_std/(500**(1/2)), g_sample_mean + 0.975*g_sample_std/500**((1/2))]

# Доверительный интервал для дисперсии
g_confidence_interval_variance = ((len(grouped_sample) - 1) * g_sample_std ** 2 / stats.chi2.ppf(0.975, len(grouped_sample) - 1),
                               (len(grouped_sample) - 1) * g_sample_std ** 2 / stats.chi2.ppf(0.025, len(grouped_sample) - 1))




print("Доверительный интервал для среднего:", confidence_interval_mean)
print("Доверительный интервал для дисперсии:", confidence_interval_variance)
print("Interval by hand:", my_interval)
print("Assymptotic Interval by hand:", )


print("Доверительный интервал для среднего, групп:", g_confidence_interval_mean)
print("Доверительный интервал для дисперсии, групп:", g_confidence_interval_variance)
print("Interval by hand:", my_interval)
print("Assymptotic Interval by hand:", )



sample = np.random.poisson(6, 500)

# Вычисление выборочного среднего
sample_mean = np.mean(sample)

# Доверительный интервал для параметра λ
confidence_interval_lambda = stats.norm.interval(0.95, loc=sample_mean, scale=np.sqrt(sample_mean / len(sample)))

print("Доверительный интервал для λ:", confidence_interval_lambda)



# Генерация выборки объемом 500 из показательного распределения с параметром λ=9
sample = np.random.exponential(1/9, 500)

# Вычисление выборочного среднего
sample_mean = np.mean(sample)

# Доверительный интервал для параметра λ
confidence_interval_lambda = stats.norm.interval(0.95, loc=sample_mean, scale=np.sqrt(sample_mean ** 2 / len(sample)))

print("Доверительный интервал для λ:", confidence_interval_lambda)



# Генерация выборки объемом 500 из биномиального распределения с параметрами n и p
n = 100  # Количество испытаний
p = 0.25  # Вероятность успеха в одном испытании

sample0 = np.random.binomial(n, p, 500)

# Вычисление выборочного среднего и выборочной пропорции успехов
sample_mean = np.mean(sample)
sample_proportion = sample_mean / n

# Доверительный интервал для параметра p
confidence_interval_p = stats.norm.interval(0.95, loc=sample_proportion, scale=np.sqrt((sample_proportion * (1 - sample_proportion)) / n))

print("Доверительный интервал для p:", confidence_interval_p)


sns.histplot(x=sample0, bins = 100, stat='probability');
plt.show()
