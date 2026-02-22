# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')
# 1. Uniform Distribution
uniform_data = np.random.uniform(low=0.0, high=10.0, size=1000)
plt.hist(uniform_data, bins=30, color='skyblue', edgecolor='black')
plt.title('Uniform Distribution [0,10]')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()
# 2. Normal Distribution
normal_data = np.random.normal(loc=0.0, scale=1.0, size=1000)
sns.histplot(normal_data, kde=True, color='green')
plt.title('Normal Distribution (mean=0, std=1)')
plt.xlabel('Value')
plt.ylabel('Density')
plt.show()
# 3. Binomial Distribution
binomial_data = np.random.binomial(n=10, p=0.5, size=1000)
sns.histplot(binomial_data, discrete=True, color='orange')
plt.title('Binomial Distribution (n=10, p=0.5)')
plt.xlabel('Number of Successes')
plt.ylabel('Frequency')
plt.show()
# 4. Poisson Distribution
poisson_data = np.random.poisson(lam=3, size=1000)
sns.histplot(poisson_data, discrete=True, color='red')
plt.title('Poisson Distribution (λ=3)')
plt.xlabel('Number of Events')
plt.ylabel('Frequency')
plt.show()
# 5. Descriptive Stats for Normal Distribution
mean = np.mean(normal_data)
std_dev = np.std(normal_data)
print('Normal Distribution Mean:', mean)
print('Standard Deviation:', std_dev)