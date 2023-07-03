from tqdm import tqdm
import numpy as np
import pandas as pd
from statsmodels.stats.rates import test_poisson, confint_poisson
from scipy.stats import poisson, gamma

def poisson_ecdf(N_obs, N_exp, density = 1000, threshold=2):
    # create l (normalised rate of mutation) in grid array between 0 and threshold
    l_ = np.linspace(0, threshold, num = int(threshold * density))
    pmf = poisson.pmf(N_obs, N_exp * l_) * N_exp / density
    cdf_norm = np.cumsum(pmf) / np.sum(pmf)
    return l_, cdf_norm


def poisson_ecdf_interval(N_obs, N_exp, cdf_density = 1000, cdf_threshold=2, alpha = 0.1):
    if N_exp == 0:
        return (None, None, None, None)
    else:
        oe = N_obs/ N_exp
        
        x, cdf = poisson_ecdf(N_obs, N_exp, density=cdf_density, threshold=cdf_threshold)
        
        if N_obs == 0:
            lower = 0
        else:
            lower = x[np.argmax(cdf[cdf < alpha/2])]
        
        if len(cdf[cdf < 1-alpha/2]) > 0:
            upper = x[np.argmax(cdf[cdf < 1-alpha/2])]
        else:
            upper = cdf_threshold

        pval = 1 - np.min(cdf[x>1])

        return oe, lower, upper, pval


def gamma_interval(count, exposure, alpha):
    rate = count/exposure
    quantile = lambda count, a: gamma(count).ppf(a) / exposure
    if count == 0:
        lower = 0
    else:
        lower = quantile(count, alpha/2)
    upper = quantile(count+1, 1-alpha/2)
    return rate, lower, upper


def poisson_exact_interval(obs, exp, alpha=0.1):
    if exp == 0:
        return (None, None, None, None)
    else:
        test = test_poisson(
            count = obs, 
            nobs=1,
            value=exp,
            method="exact-c",
            alternative = "smaller"
        )
        ci = confint_poisson(
            count = obs, 
            exposure=exp,
            method="exact-c",
            alpha = alpha
        )
    return (obs/exp, ci[0], ci[1], test.pvalue)


def estimate_constraint(input_df: pd.DataFrame, obs_col:str, exp_col:str, method = 'exact', alpha = 0.1, kw={}) -> pd.DataFrame:
    obs = input_df[obs_col]
    exp = input_df[exp_col]
    N = input_df.shape[0]
    if obs.isna().any() or exp.isna().any():
        raise ValueError('input contains NA!')
    else:
        output = np.zeros((N, 4))
        for i, (obs_, exp_) in tqdm(enumerate(zip(obs, exp)), total = N):
            if method == 'exact':
                output[i, :] = poisson_exact_interval(obs_, exp_,alpha=alpha)
            elif method == 'ecdf':
                output[i, :] = poisson_ecdf_interval(obs_, exp_, alpha=alpha, **kw)
            else:
                raise ValueError('Method required!')

        output = pd.DataFrame(output, columns = ['oe','oelf','oeuf','pval'])
        return output


    


