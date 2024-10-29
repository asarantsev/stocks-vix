import pandas
import numpy 
from matplotlib import pyplot as plt
from scipy import stats
from scipy.special import gamma
from statsmodels.api import OLS
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.stattools import adfuller
import VarGamma

def plots(data, label):
    plot_acf(data, zero = False)
    plt.title(label + '\n ACF for Original Values')
    plt.show()
    plot_acf(abs(data), zero = False)
    plt.title(label + '\n ACF for Absolute Values')
    plt.show()
    qqplot(data, line = 's')
    plt.title(label + '\n Quantile-Quantile Plot vs Normal')
    plt.show()
    
def sumSquares(data):
    data = numpy.array(data)
    return round(numpy.dot(data, data), 4)

def analysis(data, key):
    print(key)
    print('Skewness and kurtosis = ', round(stats.skew(data), 3), round(stats.kurtosis(data), 3))
    print('ACF sum of squares for original and absolute values = ', sumSquares(acf(data, nlags = 5)[1:]), sumSquares(acf(abs(data), nlags = 5)[1:]))
    
DF = pandas.read_excel('vix-stocks.xlsx', sheet_name = 'data')
vix = DF['VIX'].values
plt.plot(vix)
plt.title('VIX')
plt.show()
N = len(vix)
stocks = ['Small Total', 'Large Total', 'Small Price', 'Large Price']

print('ADF test p = ', round(adfuller(vix, maxlag = 5)[1], 3))
Heston = stats.linregress(vix[:-1], numpy.diff(vix))
print('Heston Autoregression VIX')
print('Slope = ', round(Heston.slope, 3), 'Intercept = ', round(Heston.intercept, 3), 'R^2 = ', round(Heston.rvalue**2, 3), 'p-value = ', Heston.pvalue)
Heston_res = numpy.array([vix[k+1] - vix[k] * (Heston.slope + 1) - Heston.intercept for k in range(N-1)])
plots(Heston_res, 'Heston Model')
print('\n\n\n')

lvix = numpy.log(vix)
print('ADF test p = ', round(adfuller(lvix, maxlag = 5)[1], 3))
logHeston = stats.linregress(lvix[:-1], numpy.diff(lvix))
print('Autoregression Log Heston')
print('Slope = ', round(logHeston.slope, 3), 'Intercept = ', round(logHeston.intercept, 3), 'R^2 = ', round(logHeston.rvalue**2, 3), 'p-value = ', logHeston.pvalue)
vixres = numpy.array([lvix[k+1] - lvix[k] * (logHeston.slope + 1)- logHeston.intercept for k in range(N-1)])
analysis(vixres, 'Residuals AR log VIX')
plots(vixres, ' AR log VIX')
plt.hist(numpy.exp(vixres))
plt.show()
sortRes = sorted(vixres)
loc, sigma, theta, nu = VarGamma.fit(vixres)
print('Variance-Gamma Parameters')
print('location = ', round(loc, 4))
print('sigma = ', round(sigma, 4))
print('theta = ', round(theta, 4))
print('nu = ', round(nu, 4))
print('MGF is defined for t between ', (-theta*nu - numpy.sqrt(theta**2*nu**2 + 2*sigma**2*nu))/(sigma**2*nu), ' and ', (-theta*nu + numpy.sqrt(theta**2*nu**2 + 2*sigma**2*nu))/(sigma**2*nu))
simZ = numpy.random.normal(0, 1, N-1)
simG = numpy.random.gamma(1/nu, 1, N-1)
sim = [loc + theta*nu*simG[k] + sigma*numpy.sqrt(simG[k]*nu)*simZ[k] for k in range(N-1)]
values = numpy.linspace(min(vixres), max(vixres), N-1)
plt.plot(sortRes, sorted(sim), 'o')
plt.plot(values, values, 'r')
plt.title('Quantile-Quantile Plot')
plt.show()
plt.plot(numpy.array([VarGamma.cdf(item, loc, sigma, theta, nu) for item in sortRes]))
plt.plot(numpy.linspace(0, 1, N-1))
plt.title('Probability-Probability Plot')
plt.show()
print('\n\n\n')

for key in stocks:
    print(key)
    caption = key + ' Returns'
    returns = DF[key].values
    plt.plot(returns)
    plt.title(caption)
    plt.show()
    print('Correlation of returns and volatility = ', stats.pearsonr(returns, vix)[0])
    plots(returns, caption + ' Before Normalizing')
    analysis(returns, caption + ' Before Normalizing')
    print('\n\n\n')
    nreturns = returns/vix
    plots(nreturns, caption + ' After Normalizing')
    analysis(nreturns, caption + ' After Normalizing')
    print('Mean and standard deviation of normalized returns = ', round(numpy.mean(nreturns), 4), round(numpy.std(nreturns), 4))
    print('Correlation of normalized returns and autoregression innovations = ', round(stats.pearsonr(nreturns[1:], vixres)[0], 2))
    print('\n\n\n')
    plt.plot(nreturns[1:], vixres, 'go')
    plt.title(caption + ' VIX Residuals vs Normalized Returns')
    plt.show()
    RegDF = pandas.DataFrame({'const' : 1/vix, 'vix' : 1})
    Reg = OLS(nreturns, RegDF).fit()
    print(Reg.summary())
    res = Reg.resid
    plots(res, caption + ' Regression Residuals')
    analysis(res, caption + ' Regression Residuals')
    plt.plot(res[1:], vixres, 'go')
    plt.title(caption + ' VIX Residuals vs Regression Residuals')
    plt.show()
    print('Standard deviation of residuals = ', round(numpy.std(res), 4))
    print('\n\n\n')
        
def rightHill(data, cutoff):
    return 1/(numpy.mean(data[-cutoff:]) - data[-cutoff-1])

def leftHill(data, cutoff):
    return -1/(numpy.mean(data[:cutoff]) - data[cutoff])

sortRes = sorted(vixres)
allRightHill = [rightHill(sortRes, cutoff) for cutoff in numpy.arange(30, 101)]
allLeftHill = [leftHill(sortRes, cutoff) for cutoff in numpy.arange(30, 101)]
plt.plot(numpy.arange(30, 101), allRightHill)
plt.title('Right-tail Hill estimator')
plt.show()
plt.plot(numpy.arange(30, 101), allLeftHill)
plt.title('Left-tail Hill estimator')
plt.show()