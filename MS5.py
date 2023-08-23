# -*- coding: windows-1251 -*-
import numpy as np
from scipy.stats import norm, kstest, chi2
import matplotlib.pyplot as plt

# ��������� �������
np.random.seed(1)
sample = np.random.normal(5, 8, 1000)

# �������� �������� ����������� �� �������� �������
observed, bins = np.histogram(sample, bins='auto')
expected = len(sample) * np.diff(bins) * norm.pdf(bins[:-1], loc=sample.mean(), scale=sample.std())
chi2_stat = np.sum((observed - expected)**2 / expected)
p_value_chi2 = 1 - chi2.cdf(chi2_stat, len(observed) - 1)
print("�������� ���������� ��-������� �������:", chi2_stat)
print("�������� p-�������� �� ��-������� �������:", p_value_chi2)

# �������� �������� ����������� �� �������� �����������
ks_stat, p_value_ks = kstest(sample, 'norm', args=(sample.mean(), sample.std()))
print("�������� ���������� �����������:", ks_stat)
print("�������� p-�������� �� �����������:", p_value_ks)


#from scipy.stats import norm, kstest, chi2

# ��������� "�����������" �������
np.random.seed(1)
sample = np.random.normal(5, 8, 1000)
epsilon = np.random.uniform(-2, 3, 1000)
sample_noisy = sample + epsilon

# �������� �������� ����������� �� �������� �������
observed, bins = np.histogram(sample_noisy, bins='auto')
expected = len(sample_noisy)* np.diff(bins) * norm.pdf(bins[:-1], loc=sample_noisy.mean(), scale=sample_noisy.std())
chi2_stat = np.sum((observed - expected)**2 / expected)
p_value_chi2 = 1 - chi2.cdf(chi2_stat, len(observed) - 1) 
print('\n ����������� ���:')
print("�������� ���������� ��-������� �������:", chi2_stat)
print("�������� p-�������� �� ��-������� �������:", p_value_chi2)

# �������� �������� ����������� �� �������� �����������
ks_stat, p_value_ks = kstest(sample_noisy, 'norm', args=(sample_noisy.mean(), sample_noisy.std()))
print("�������� ���������� �����������:", ks_stat)
print("�������� p-�������� �� �����������:", p_value_ks)

#plt.hist(sample_noisy, bins=100, alpha=0.7, label='Uniform Noised', density=True)
#x = np.linspace(-30, 40, 100)
#pdf = norm.pdf(x, loc=5, scale=8)
#plt.plot(x, pdf, 'r', label='Normal Distribution')

## ��������� ���� � �������
#plt.xlabel('��������')
#plt.ylabel('�������')
#plt.legend()

## ����������� �������
#plt.show()


from scipy.stats import norm, kstest, chi2, cauchy

# ��������� "�����������" �������
np.random.seed(1)
sample = np.random.normal(5, 8, 1000)
epsilon1 = np.random.standard_cauchy(1000) #
epsilon = cauchy.rvs(scale = 0.02, size=1000)
sample_noisy = sample + epsilon

# �������� �������� ����������� �� �������� �������

observed, bins = np.histogram(sample_noisy, bins='auto')
expected = len(sample_noisy)* np.diff(bins) * norm.pdf(bins[:-1], loc=sample_noisy.mean(), scale=sample_noisy.std())
chi2_stat = np.sum((observed - expected)**2 / expected)
p_value_chi2 = 1 - chi2.cdf(chi2_stat, len(observed) - 1)


#observed, _ = np.histogram(sample_noisy, bins='auto')
#expected = len(sample_noisy) / len(observed) * norm.pdf(np.linspace(sample_noisy.min(), sample_noisy.max(), len(observed)+1), loc=sample_noisy.mean(), scale=sample_noisy.std())
#chi2_stat, p_value_chi2 = chisquare(observed, expected)
print('\n ���� ���:')
print("�������� ���������� ��-������� �������:", chi2_stat)
print("�������� p-�������� �� ��-������� �������:", p_value_chi2)

# �������� �������� ����������� �� �������� �����������
ks_stat, p_value_ks = kstest(sample_noisy, 'norm', args=(sample_noisy.mean(), sample_noisy.std()))
print("�������� ���������� �����������:", ks_stat)
print("�������� p-�������� �� �����������:", p_value_ks)




import numpy as np
from scipy.stats import ks_2samp

# ��������� �������
np.random.seed(1)
X1 = np.random.normal(5, 8, 100)
X2 = np.random.normal(3, 5, 100)
Xmod = np.concatenate((X1[:50], X2[50:]))

# �������� �������� ������������ ������������� �� �������� �����������-��������
ks_stat, p_value_ks = ks_2samp(X1, X2)
ks_stat_m, p_value_ks_m = ks_2samp(Xmod, X2)
print('\n ������������:')
print("�������� ���������� �����������-��������:", ks_stat)
print("�������� p-�������� �� �����������-��������:", p_value_ks)

print("�������� ���������� �����������-��������:", ks_stat_m)
print("�������� p-�������� �� �����������-��������:", p_value_ks_m)
