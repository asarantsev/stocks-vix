import numpy as np
from scipy import stats
import pandas as pd
from statsmodels.tsa.stattools import acf

DF = pd.read_excel('VIX.xlsx', sheet_name = 'data')
vix = DF['VIX'].values
N = len(vix)

lvix = np.log(vix)
lam = stats.yeojohnson(lvix)[1]
print('lambda = ', lam)
tvix = stats.yeojohnson(lvix)[0]

print('Shapiro-Wilk p = ', stats.shapiro(tvix)[1])
print('Jarque-Bera p = ', stats.jarque_bera(tvix)[1])

Reg = stats.linregress(tvix[:-1], tvix[1:])
resid = tvix[1:] - Reg.slope * tvix[:-1] - Reg.intercept * np.ones(N - 1)
print(Reg)
print('ACF for original residuals')
print(acf(resid, nlags = 10, qstat = True)[2])
print('ACF for absolute values')
print(acf(abs(resid), nlags = 10, qstat = True)[2])
print('Shapiro-Wilk p = ', stats.shapiro(resid)[1])
print('Jarque-Bera p = ', stats.jarque_bera(resid)[1]) 