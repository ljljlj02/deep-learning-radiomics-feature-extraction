import pandas as pd
from scipy.stats import chi2_contingency
import scipy.stats
from scipy.stats import chi2

df = pd.read_csv("xxx.csv")
table = pd.crosstab(df['group'],df['xxx'])
result = scipy.stats.chi2_contingency(table,correction=False)
# print('result:',result)
expected_freq = result[3]
degree_of_freedom = result[2]
p_value = result[1]
chi_static = result[0]
observed_freq = table.values
alpha=0.05
#critical_value
critical_value=chi2.ppf(q=1-alpha,df=degree_of_freedom)
print('critical_value:',critical_value)
print('chi_static,p_value',chi_static,p_value)


