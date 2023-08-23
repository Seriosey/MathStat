# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import random as rn


data = []

for x in range(145): data.append(rn.uniform(50,54.999999))
for x in range(310): data.append(rn.uniform(55,59.999999))
for x in range(305): data.append(rn.uniform(60,64.999999))
for x in range(120): data.append(rn.uniform(65,69.999999))
for x in range(20): data.append(rn.uniform(70,74.999999))

#print(y)
#plt.hist(y, color = 'blue', edgecolor = 'black', bins = int(25/5)) 

table = np.array([[52.5, 145], [57.5, 310], [62.5, 305], [67.5, 120], [72.5, 20]])

def average(table):
    av=0
    for cell in table:
        av += cell[0]*cell[1]
    return av/900


def math_expection(table):
    me=0
    for cell in table:
        me += cell[0]*cell[1]
    return me/900


def dispersion(_table):
    disp = 0
    av = average(_table)
    for cell in _table:
        disp+= cell[1]*(cell[0]- av)**2

    return disp/900

def cenral_moment(_table, k):
    me = math_expection(_table)
    diff_table = np.copy(_table)
    for cell in diff_table:
        cell[0] = (cell[0] - me)**k
    #print(diff_table)
    return math_expection(diff_table)

def skewness(_table):
    return cenral_moment(_table, 3)/dispersion(_table)**(3/2)


def excess(_table):
    return cenral_moment(_table, 4)/dispersion(_table)**(2) - 3

def variation(_table):
    return (dispersion(_table))**(1/2)/average(_table)


mean = np.mean(data)
variance = np.var(data)
std_dev = np.std(data)
median = np.median(data)
skewness_true = np.mean(((data - mean)/std_dev)**3)
kurtosis = np.mean(((data - mean)/std_dev)**4) - 3

variation_true =  np.std(data, ddof= 1 ) / np.mean (data)



print('true characteristics:')
print('Mean: ', mean)
print('Variation: ',variation_true)
print('Std_dev: ',std_dev)
print('Skewness: ',skewness_true)
print('Kurtosis: ',kurtosis)

# генерация случайной выборки с нормальным распределением для добавления шума
noise = np.random.normal(loc=0, scale=1000, size=len(data))

# добавление шума к исходным данным
noisy_data = data + noise

# вычисление статистических параметров для шумовых данных
noisy_mean = np.mean(noisy_data)
noisy_variance = np.std(noisy_data, ddof= 1 ) / np.mean (noisy_data)
noisy_std_dev = np.std(noisy_data)
noisy_median = np.median(noisy_data)
noisy_skewness = np.mean(((noisy_data - noisy_mean)/noisy_std_dev)**3)
noisy_kurtosis = np.mean(((noisy_data - noisy_mean)/noisy_std_dev)**4) - 3
variation_noisy_true =  np.std(noisy_data, ddof= 1 ) / np.mean (noisy_data)


print('true noised characteristics:')
print('Mean: ', noisy_mean)
print('Variation: ',noisy_variance)
print('Std_dev: ',(noisy_std_dev)**2)
print('Skewness: ',noisy_skewness)
print('Kurtosis: ',noisy_kurtosis)
print(noisy_median)

x = np.linspace(50,75)

f = np.sort(data)
t = np.arange(len(f))/float(len(f))

f_noisy = np.sort(noisy_data)
plt.hist(f_noisy, color = 'blue', edgecolor = 'black', bins = int(1000/5))
#plt.plot(f_noisy, t)

#plt.plot(x, y)
plt.show()

print(average(table))
print('\n')
print(variation(table))
print('\n')
print((dispersion(table))**(1/2))
print('\n')
#print(cenral_moment(table, 3))
#print('\n')
print(skewness(table))
print('\n')
print(excess(table))
