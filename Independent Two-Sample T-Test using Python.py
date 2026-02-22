from scipy import stats
import numpy as np

A = np.array([78,84,92,88,75,80,85,90,87,79,78,84,92,88,75,80,85,90,87,79])
B = np.array([82,88,75,90,78,85,88,77,92,80,82,88,75,90,78,85,88,77,92,80])

t_val, p_val = stats.ttest_ind(A, B)
alpha = 0.05
df = len(A) + len(B) - 2
crit_t = stats.t.ppf(1 - alpha/2, df)

print("T-value:", t_val)
print("P-Value:", p_val)
print("Critical t-value:", crit_t)

print('T-test Result:')
if np.abs(t_val) > crit_t:
    print('Significant difference found.')
else:
    print('No significant difference.')

print('P-test Result:')
if p_val <= alpha:
    print('Reject H0. Significant difference found.')
else:
    print('Fail to reject H0. No strong evidence of difference.')