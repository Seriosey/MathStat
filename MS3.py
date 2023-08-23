# -*- coding: windows-1251 -*-
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.neighbors import KernelDensity

x = np.linspace(-20, 30, 1000)

# Гауссовское распределение N(5, 8)
#np.random.seed(0)
sample_g = np.random.normal(5, 8, 500)

# Равномерное распределение на отрезке [5, 8]
sample_uniform = np.random.uniform(5, 8, 500)

# Распределение Коши
sample_cauchy0 = np.random.standard_cauchy(500)
sample_cauchy = stats.cauchy.pdf(x, 5, 4)
true_density_cauchy = 4 / (np.pi * ((x - 3) ** 2 + 16))


def Approx(sample):

    h_silverman_gaussian = 1.06 * np.std(sample) * (len(sample) ** (-1 / 5))

    kde_g = KernelDensity(bandwidth = h_silverman_gaussian, kernel='gaussian')
    kde_g.fit(sample[:, np.newaxis])

    kde_u = KernelDensity(bandwidth = h_silverman_gaussian, kernel='tophat')
    kde_u.fit(sample[:, np.newaxis])

    kde_t = KernelDensity(bandwidth = h_silverman_gaussian, kernel='linear')
    kde_t.fit(sample[:, np.newaxis])

    # Генерируем точки для построения плотности
    x = np.linspace(min(sample), max(sample), 1000)
    log_density_g = kde_g.score_samples(x[:, np.newaxis])
    density_g = np.exp(log_density_g)

    log_density_u = kde_u.score_samples(x[:, np.newaxis])
    density_u = np.exp(log_density_u)

    log_density_t = kde_t.score_samples(x[:, np.newaxis])
    density_t = np.exp(log_density_t)

    # Построение гистограммы
    plt.hist(sample, bins= 50, density=True, alpha=0.7, label='Histogram')

    # Построение функции плотности
    plt.plot(x, density_g, color='red', label='Gaussian Kernel Density Estimation')
    plt.plot(x, density_u, color='purple', label='Uniform Kernel Density Estimation')
    plt.plot(x, density_t, color='pink', label='Triangle Kernel Density Estimation')
    plt.xlabel('Значение')
    plt.ylabel('Плотность')
    plt.title('Ядерная оценка плотности')


Approx(sample_cauchy)


# Построение графиков

plt.legend(loc="upper right")
plt.show()

