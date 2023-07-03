import numpy as np
import pandas as pd
from scipy.stats import poisson

def oe_poisson_pmf(N_obs, N_exp, density = 1000, threshold=3):
    # create l (normalised rate of mutation) in grid array between 0 and threshold
    l_ = np.linspace(0, threshold, num = int(threshold * density))
    pmf = poisson.pmf(N_obs, N_exp * l_) * N_exp / density
    return l_, pmf

# def oe_normalised_pmf(N_obs, N_exp, density = 1000, threshold=3):
#     l, pmf = oe_poisson_pmf(N_obs, N_exp, density = density, threshold=threshold)
#     pmf_sum = np.sum(pmf) / 

def oe_adjusted_cdf(l, pmf):
    # Sum of probility mass for rates up to l
    cdf_ = np.cumsum(pmf)
    if np.max(cdf_) < 0.9:
        print('Warning: poor coverage of CI')
    cdf_norm_ = cdf_ / np.max(cdf_)
    return cdf_norm_

def oe_confidence_interval(l, cdf, alpha = 0.9):
    alpha = (1 - alpha) / 2
    lower = l[np.min(np.where(cdf > alpha))]
    upper = l[np.max(np.where(cdf < 1 - alpha))]
    return (lower, upper)

def oe_log_pval(l, cdf, baseline = 1):
    return np.log(1 - cdf[np.min(np.where(l>=1))])

def random_region_nposs(length):
    r = np.random.randint(0,count_missense_by_loc.aa_pos_start.max())
    return count_missense_by_loc[
        (count_missense_by_loc.aa_pos_start >= r) &
        (count_missense_by_loc.aa_pos_start < r + 10)
        ].n_mut.sum()

def sensitivity(N_exp, constraint):
    N_obs = np.round(N_exp * constraint)
    #N_exp = N_obs / constraint
    l0, pmf =  oe_poisson_pmf(N_obs, N_exp)
    cdf_norm = oe_adjusted_cdf(l0, pmf)
    lower, upper = oe_confidence_interval(l0, cdf_norm)
    pval = oe_log_pval(l0, cdf_norm)
    return {
        'N_exp':N_exp,
        'N_obs':N_obs,
        'l':l0,
        'pmf':pmf,
        'cdf':cdf_norm,        
        'oe':N_obs/N_exp,
        'oe_lower':lower,
        'oe_upper':upper,
        'pval':pval
        }

def sensitivity_at_constraint(constraint, N_obs_max):
    output = pd.DataFrame({
        'n_obs':np.arange(1,N_obs_max+1)
    })
    output['n_exp'] = output['n_obs'] / constraint
    output['n_poss'] = output['n_exp'] / 0.09
    output['region_size'] = output['n_poss'] / 6.5
    output['sensitivity'] = output.n_obs.map(lambda n: sensitivity(n, constraint))
    output = pd.concat((output.drop(columns=['sensitivity']), pd.json_normalize(output.sensitivity.values)),axis=1)
    return output


def plot_sensitivity(constraint,region_size_max,axs):
    output = sensitivity_at_constraint(constraint,region_size_max)
    axs.plot(output['region_size'],output['oe_upper'],color='k',linestyle='dashed',label='Upper bound of CI')
    axs.plot(output['region_size'],output['oe'],color = 'k',label='True value')
    axs.hlines(y=1,xmin=output['region_size'].min(),xmax=output['region_size'].max(),color='r',linestyle='dotted',label='Threshold')
    axs.set_ylabel('observed/expected')
    axs.set_xlabel('Size of region')
    return axs

def crossover(constraint,region_size_max):
    output = sensitivity_at_constraint(constraint,region_size_max)
    return output[output.oe_upper < 1].region_size.min()


def oe_CI(obs,exp,threshold=3):
    lower = []
    upper = []
    for N_obs, N_exp in zip(obs,exp):
        l0, pmf =  oe_poisson_pmf(N_obs, N_exp,threshold=threshold)
        cdf_norm = oe_adjusted_cdf(l0, pmf)
        lower_, upper_ = oe_confidence_interval(l0, cdf_norm)
        lower.append(lower_)
        upper.append(upper_)
    return (np.array(lower), np.array(upper))

         
def oe_bounds(obs,exp):
    oe = obs/exp
    oe_lower, oe_upper = oe_CI(obs, exp)
    oe_err_lower = oe - oe_lower
    oe_err_upper = oe_upper - oe
    df = pd.DataFrame(dict(
        oe= oe,
        oe_lower = oe_lower,
        oe_upper = oe_upper,
        oe_err_lower = oe_err_lower,
        oe_err_upper = oe_err_upper
    ))
    return df

def constraint_by_decile(obs, exp, prediction):
    decile =  pd.qcut(prediction,10,duplicates='drop').astype(str)
    df = pd.DataFrame(dict(obs=obs.values,exp=exp.values,decile=decile))
    df = df.groupby(decile).agg(obs=('obs','sum'),exp=('exp','sum')).reset_index()
    df[['oe', 'oe_lower','oe_upper','oe_err_lower','oe_err_upper']] = oe_bounds(df.obs,df.exp)
    return df

def plot_constraint_by_decile(constraint_by_decile_, ax):
    ax.errorbar(
        x=constraint_by_decile_.index,
        y=constraint_by_decile_.oe,
        yerr=np.stack((
            constraint_by_decile_.oe_err_lower,
            constraint_by_decile_.oe_err_upper
        )),    
    marker='o',linestyle='none'
    )

    ax.set_xticks(constraint_by_decile_.index)
    return ax

def plot_constraint_by_rank(df, col_name, nonsense_obs = 0, nonsense_exp = 0,ax = None):
    if ax is None:
        ax = plt.gca()
    n = []
    oe = []
    lower = []
    upper = []

    for x in range(0,80):
        df = df.sample(frac=1)
        included = df[df[col_name].rank(pct=True,method='first') > x/100]
        n_inc = included.shape[0]
        N_obs, N_exp = included[['observed_variants','expected_variants']].sum()
        l0, pmf =  oe_poisson_pmf(N_obs+nonsense_obs, N_exp+nonsense_exp,threshold=8)
        cdf_norm = oe_adjusted_cdf(l0, pmf)
        lower_, upper_ = oe_confidence_interval(l0, cdf_norm)
        n.append(n_inc)
        oe.append((N_obs+nonsense_obs)/(N_exp+nonsense_exp))
        lower.append(lower_)
        upper.append(upper_)
    print(min(upper))
    ax.plot(n, oe)
    ax.plot(n, lower)
    ax.plot(n, upper)