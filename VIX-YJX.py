import numpy as np
from scipy import stats
import pandas as pd
from statsmodels.tsa.stattools import acf

DF = pd.read_excel('VIX.xlsx', sheet_name = 'data')
vix = DF['VIX'].values
lvix = np.log(vix)
lam = stats.yeojohnson(lvix)[1]
tvix = stats.yeojohnson(lvix)[0]
Reg = stats.linregress(tvix[:-1], tvix[1:])
resid = tvix[1:] - Reg.slope * tvix[:-1] - Reg.intercept * np.ones(479)
print(acf(resid, nlags = 10, qstat = True))
print(acf(abs(resid), nlags = 10, qstat = True))
