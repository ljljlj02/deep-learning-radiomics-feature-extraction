from scipy.stats import ranksums
import pandas as pd
import math
# a1= [70, 80, 72, 76, 76, 76, 72, 78, 82, 92, 68, 84]
# b1= [68, 72, 62, 70, 66, 68, 52, 64]
df = pd.read_csv('/xxx.csv', usecols=[0,1], header=0)
a=df['LGG'].tolist()
a2=list(map(int, a))
b=df['HGG'].tolist()
b = [x for x in b if math.isnan(x) == False]
b2=list(map(int, b))
#--------------Wilcoxon rank-sum test
result=ranksums(a2, b2)
print('result:',result)
#--------------Wilcoxon signed-rank test
# w, p = wilcoxon(a1, b1)
# print('w,p:',w,p)


