import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Example dataset
df = pd.DataFrame({ 'Category': ['A', 'B', 'C', 'D'], 'Values': [23, 45, 56, 78] })

# Bar Chart
sns.barplot(x='Category', y='Values', data=df)

plt.title('Bar Chart')
plt.show()

# Line Plot
plt.plot(df['Category'], df['Values'])
plt.title('Line Plot')
plt.show()
# Scatter Plot
plt.scatter(df.index, df['Values'])
plt.title('Scatter Plot')
plt.show()