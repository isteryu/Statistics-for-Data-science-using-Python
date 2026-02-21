import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Sample Data
data = [10, 20, 20, 30, 40, 50, 60, 70, 80, 90]


# Descriptive Statistics
print('Mean:', np.mean(data))
print('Median:', np.median(data))
print('Mode:', stats.mode(data))
print('Standard Deviation:', np.std(data))
print('Variance:', np.var(data))

# Visualization
sns.histplot(data, kde=True)
plt.title('Histogram')
plt.show()
sns.boxplot(data=data)
plt.title('Boxplot')
plt.show()
x = np.arange(len(data))
plt.scatter(x, data)
plt.title('Scatter Plot')
plt.xlabel('Index')
plt.ylabel('Values')
plt.show()